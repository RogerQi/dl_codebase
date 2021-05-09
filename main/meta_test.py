import __init_lib_path
from config_guard import cfg, update_config_from_yaml
import dataset
import backbone
import classifier
import loss
import utils

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm, trange

from IPython import embed

def parse_args():
    parser = argparse.ArgumentParser(description = "Roger's Deep Learning Playground")
    parser.add_argument('--cfg', help = "specify particular yaml configuration to use", required = True,
        default = "configs/mnist_torch_official.taml", type = str)
    parser.add_argument('--load', help="specify saved checkpoint to evaluate", required=True, type=str)
    args = parser.parse_args()

    return args

# A generalized imshow helper function which supports displaying (CxHxW) tensor
def generalized_imshow(cfg, arr):
    if isinstance(arr, torch.Tensor) and arr.shape[0] == 3:
        ori_rgb_np = np.array(arr.permute((1, 2, 0)).cpu())
        if 'normalize' in cfg.DATASET.TRANSFORM.TEST.transforms:
            rgb_mean = cfg.DATASET.TRANSFORM.TEST.TRANSFORMS_DETAILS.NORMALIZE.mean
            rgb_sd = cfg.DATASET.TRANSFORM.TEST.TRANSFORMS_DETAILS.NORMALIZE.sd
            ori_rgb_np = (ori_rgb_np * rgb_sd) + rgb_mean
        assert ori_rgb_np.max() <= 1.1, "Max is {}".format(ori_rgb_np.max())
        ori_rgb_np[ori_rgb_np >= 1] = 1
        arr = (ori_rgb_np * 255).astype(np.uint8)
    plt.imshow(arr)
    plt.show()

def meta_test_one(cfg, backbone_net, post_processor, criterion, feature_shape, device, support_img_tensor_bchw, support_mask_tensor_bhw,
                            query_img_tensor_bchw, query_mask_tensor_bhw):
    tune_whole_arch = False
    tune_cosine_head_only = True
    post_processor.replace(cfg, feature_shape)
    post_processor.pixel_classifier.to(device)
    num_shots = support_img_tensor_bchw.shape[0]
    # TODO: maybe there is a better way to initialize?
    # post_processor.pixel_classifier.class_mat.reset_parameters()
    with torch.no_grad():
        feature, aux_dict = backbone_net(support_img_tensor_bchw)
        mask_logit = post_processor.logit_forward(feature)
        downsampled_mask = F.interpolate(support_mask_tensor_bhw.view((num_shots, 1, 417, 417)).float(),
            size = (14, 14), mode = "bilinear", align_corners=False)
        # Use aggresive nonzero selection
        fg_channel_idx = (downsampled_mask > 0).view((-1,))
        bg_channel_idx = torch.logical_not(fg_channel_idx)
        channel_logit = mask_logit.permute((1, 0, 2, 3)).reshape(4096, -1)
        fg_class_vec = channel_logit[:, fg_channel_idx]
        fg_class_vec = torch.mean(fg_class_vec, dim = 1).reshape((1, 4096, 1, 1))
        bg_class_vec = channel_logit[:, bg_channel_idx]
        bg_class_vec = torch.mean(bg_class_vec, dim = 1).reshape((1, 4096, 1, 1))
        bg_fg_class_mat = torch.cat([bg_class_vec, fg_class_vec])
        post_processor.pixel_classifier.class_mat.weight.data = bg_fg_class_mat
        # Average may work better?
        # map_logits = downsampled_mask * mask_logit
    # Default init
    # torch.nn.init.kaiming_normal_(post_processor.pixel_classifier.class_mat.weight)
    post_processor.train()

    optimizer = optim.SGD(post_processor.parameters(), lr = 1e-5, momentum = 0.9, dampening = 0.9,
                            weight_decay = 0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [30, 50, 60], gamma = 0.1)

    max_epoch = 20
    for epoch in range(1, max_epoch + 1):
        # tune HEAD weights
        prv_fc6_weight = post_processor.fc6.weight.clone().detach()
        prv_cos_weight = post_processor.pixel_classifier.class_mat.weight.clone().detach()
        if tune_whole_arch:
            assert not tune_cosine_head_only
            feature, aux_dict = backbone_net(support_img_tensor_bchw)
            mask_logit = post_processor.logit_forward(feature)
        else:
            if tune_cosine_head_only:
                with torch.no_grad():
                    feature, aux_dict = backbone_net(support_img_tensor_bchw)
                    mask_logit = post_processor.logit_forward(feature).detach()
            else:
                with torch.no_grad():
                    feature, aux_dict = backbone_net(support_img_tensor_bchw).detach()
                mask_logit = post_processor.logit_forward(feature)
        ori_spatial_res = support_img_tensor_bchw.shape[-2:]
        predicted_mask = post_processor.cosine_forward(mask_logit, ori_spatial_res)
        loss = criterion(predicted_mask, support_mask_tensor_bhw)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        new_fc6_weight = post_processor.fc6.weight.clone().detach()
        new_cos_weight = post_processor.pixel_classifier.class_mat.weight.clone().detach()
        assert not (prv_cos_weight == new_cos_weight).all()
        if tune_cosine_head_only:
            assert (prv_fc6_weight == new_fc6_weight).all()
        else:
            assert not (prv_fc6_weight == new_fc6_weight).all()

        # Train Eval
        pred_map = predicted_mask.max(dim = 1)[1]
        pixel_acc_list = []
        iou_list = []
        precision_list = []
        recall_list = []
        for i in range(pred_map.shape[0]):
            batch_acc, _ = utils.compute_pixel_acc(pred_map[i], support_mask_tensor_bhw[i], fg_only=cfg.METRIC.SEGMENTATION.fg_only)
            iou = utils.compute_iou(
                np.array(pred_map[i].cpu()),
                np.array(support_mask_tensor_bhw[i].cpu(), dtype=np.int64),
                cfg.meta_testing_num_classes,
                fg_only=cfg.METRIC.SEGMENTATION.fg_only
            )
            precision = utils.compute_binary_precision(pred_map[i].cpu().numpy(), support_mask_tensor_bhw[i].cpu().numpy())
            recall = utils.compute_binary_recall(pred_map[i].cpu().numpy(), support_mask_tensor_bhw[i].cpu().numpy())
            pixel_acc_list.append(batch_acc.cpu().numpy())
            iou_list.append(iou)
            precision_list.append(precision)
            recall_list.append(recall)
        train_pixel_acc = np.mean(pixel_acc_list)
        train_iou = np.mean(iou_list)
        # print("Zero pixel cnt: {} One pixel cnt: {}".format((pred_map == 0).sum(), (pred_map == 1).sum()))
        # print("[Epoch {}] Train loss: {:.4f} Pixel Acc: {:.4f} IoU: {:.4f} Precision: {:.4f} Recall: {:.4f}".format(epoch,
        #                                                         loss.item(), np.mean(pixel_acc_list), np.mean(iou_list),
        #                                                         np.mean(precision_list), np.mean(recall_list)))
        # Query Eval
        with torch.no_grad():
            eval_feature, aux_dict = backbone_net(query_img_tensor_bchw)
            eval_ori_spatial_res = query_img_tensor_bchw.shape[-2:]
            eval_predicted_mask = post_processor(eval_feature, eval_ori_spatial_res)
            eval_loss = criterion(eval_predicted_mask, query_mask_tensor_bhw).item()
            pred_map = eval_predicted_mask.max(dim = 1)[1]
            assert pred_map.shape[0] == 1
            batch_acc, _ = utils.compute_pixel_acc(pred_map, query_mask_tensor_bhw, fg_only=cfg.METRIC.SEGMENTATION.fg_only)
            iou = utils.compute_iou(
                np.array(pred_map[0].cpu()),
                np.array(query_mask_tensor_bhw[0].cpu(), dtype=np.int64),
                cfg.meta_testing_num_classes,
                fg_only=cfg.METRIC.SEGMENTATION.fg_only
            )
            # print("[Epoch {}] Eval loss: {:.4f} Pixel Acc: {:.4f} IoU: {:.4f}".format(epoch, eval_loss, batch_acc, iou))

        # step scheduler
        scheduler.step()
    
    return float(train_pixel_acc), float(train_iou), float(batch_acc), float(iou)

# Now this function is PASCL5i specific. When more few-shot dataset are included,
# I will generalize this method.
def meta_test(cfg, backbone_net, post_processor, feature_shape, criterion, device, meta_test_set):
    backbone_net.eval()

    # Some assertions and prepare necessary variables
    assert cfg.task.startswith('few_shot')
    assert cfg.DATASET.dataset == "pascal_5i"
    num_shots = cfg.META_TEST.shot
    assert num_shots != -1

    # Following PANet, we use five consistent random seeds.
    # For each seed, 1000 (S, Q) pairs are sampled.
    np.random.seed(1221)
    seed_list = np.random.randint(0, 99999, size = (5, ))

    # Generate indices list
    # Struct:
    #   - meta_test_task[0]: meta_test_single_run
    #   - meta_test_task[0][0]: meta_test_batch_dict
    #       - key: query_idx, support_idx_list, novel_class_id
    meta_test_task = []

    for seed in seed_list:
        np.random.seed(seed)

        # Query images are selected with replacement (support set may differ)
        meta_test_single_run = []

        query_image_list = np.random.randint(0, len(meta_test_set), size = (1000, ))
        for query_img_idx in query_image_list:
            # Get list of class in the image
            class_list = meta_test_set.dataset.get_class_in_an_image(query_img_idx)

            # Select a novel class in the list to be used
            novel_class_idx = np.random.choice(class_list)

            # Get support set
            potential_support_img = meta_test_set.dataset.get_img_containing_class(novel_class_idx)

            # Remove query image from the set
            potential_support_img.remove(query_img_idx)
            assert len(potential_support_img) >= num_shots
            support_img_idx_list = np.random.choice(potential_support_img, size = (num_shots, ), replace = False)
            
            meta_test_batch_dict = {
                "query_idx": query_img_idx,
                "support_idx_list": support_img_idx_list,
                "novel_class_id": novel_class_idx
            }

            meta_test_single_run.append(meta_test_batch_dict)
        
        # End of current seed.
        meta_test_task.append(meta_test_single_run)

    # Meta Test!    
    for i, meta_test_single_run in enumerate(meta_test_task):
        for j, meta_test_batch in tqdm(enumerate(meta_test_single_run)):
            novel_class_id = meta_test_batch['novel_class_id']
            query_img, query_target = meta_test_set[meta_test_batch['query_idx']]
            query_target_mask = (query_target == novel_class_id)
            support_img_list = []
            support_mask_list = []
            for supp_idx in meta_test_batch['support_idx_list']:
                supp_img, supp_target = meta_test_set[supp_idx]
                support_img_list.append(supp_img)
                support_mask_list.append(supp_target == novel_class_id)
            if False:
                # Change this to True to visualize image
                print("Visualizing query images...")
                generalized_imshow(cfg, query_img)
                generalized_imshow(cfg, query_target_mask)
                print("Visualizing support images...")
                for k in range(len(support_mask_list)):
                    generalized_imshow(cfg, support_img_list[k])
                    generalized_imshow(cfg, support_mask_list[k])
            # Pad tensor
            query_img_tensor_bchw = query_img.view((1, ) + query_img.shape)
            query_mask_tensor_bhw = (query_target_mask.view((1, ) + query_target_mask.shape)).long()
            support_img_tensor_bchw = torch.stack(support_img_list)
            support_mask_tensor_bhw = torch.stack(support_mask_list).long()
            train_acc, train_iou, acc, iou = meta_test_one(cfg, backbone_net, post_processor, criterion, feature_shape, device,
                            support_img_tensor_bchw.cuda(), support_mask_tensor_bhw.cuda(),
                            query_img_tensor_bchw.cuda(), query_mask_tensor_bhw.cuda())
            meta_test_task[i][j]['fg_pixel_acc'] = acc
            meta_test_task[i][j]['fg_binary_iou'] = iou
            print("[Train] Acc: {:.4f} IoU: {:.4f} [Eval] Acc: {:.4f} IoU: {:.4f}".format(train_acc, train_iou, acc, iou))
        break # only 1 run for now

    # Gather test statistics
    for i, meta_test_single_run in enumerate(meta_test_task):
        acc_list = []
        iou_list = []
        for j, meta_test_batch in enumerate(meta_test_single_run):
            acc_list.append(meta_test_task[i][j]['fg_pixel_acc'])
            iou_list.append(meta_test_task[i][j]['fg_binary_iou'])
        print("Accuracy: {} IoU: {}".format(np.mean(np.array(acc_list)), np.mean(np.array(iou_list))))

def main():
    # --------------------------
    # | Initial set up
    # |  1. Parse arg and yaml
    # |  2. Set device
    # |  3. Set seed
    # --------------------------
    args = parse_args()
    update_config_from_yaml(cfg, args)

    use_cuda = not cfg.SYSTEM.use_cpu
    device_str = "cuda" if use_cuda else "cpu"
    device = torch.device(device_str)

    kwargs = {'num_workers': cfg.SYSTEM.num_workers, 'pin_memory': cfg.SYSTEM.pin_memory} if use_cuda else {}

    torch.manual_seed(cfg.seed)

    # --------------------------
    # | Prepare datasets
    # --------------------------
    _, meta_test_set = dataset.meta_dispatcher(cfg)

    print("Meta test set contains {} data points.".format(len(meta_test_set)))

    # --------------------------
    # | Get ready to learn
    # |  1. Prepare network and loss
    # |  2. Prepare optimizer
    # |  3. Set learning rate
    # --------------------------
    backbone_net = backbone.dispatcher(cfg)
    backbone_net = backbone_net(cfg).to(device)
    feature_shape = backbone_net.get_feature_tensor_shape(device)
    print("Flatten feature length: {}".format(feature_shape))
    post_processor = classifier.dispatcher(cfg, feature_shape)
    
    post_processor = post_processor.to(device)
    
    print("Initializing backbone with pretrained weights from: {}".format(args.load))
    trained_weight_dict = torch.load(args.load, map_location=device_str)
    backbone_net.load_state_dict(trained_weight_dict["backbone"], strict=True)
    post_processor.load_state_dict(trained_weight_dict["head"], strict=True)

    criterion = loss.dispatcher(cfg)

    meta_test(cfg, backbone_net, post_processor, feature_shape, criterion, device, meta_test_set)


if __name__ == '__main__':
    main()