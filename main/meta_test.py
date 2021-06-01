import __init_lib_path
from config_guard import cfg, update_config_from_yaml
import dataset
import backbone
import classifier
import loss

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description = "Roger's Deep Learning Playground")
    parser.add_argument('--cfg', help = "specify particular yaml configuration to use", required = True,
        default = "configs/mnist_torch_official.taml", type = str)
    parser.add_argument('--load', help="specify saved checkpoint to evaluate", required=True, type=str)
    args = parser.parse_args()

    return args

def masked_average_pooling(mask_b1hw, feature_bchw):
    '''
    Params
        - mask_b1hw: a binary mask whose element-wise value is either 0 or 1
        - feature_bchw: feature map obtained from the backbone
    
    Return: Mask-average-pooled vector of shape 1 x C
    '''
    if len(mask_b1hw.shape) == 3:
        mask_b1hw = mask_b1hw.view((mask_b1hw.shape[0], 1, mask_b1hw.shape[1], mask_b1hw.shape[2]))

    # Spatial resolution mismatched. Interpolate feature to match mask size
    if mask_b1hw.shape[-2:] != feature_bchw.shape[-2:]:
        feature_bchw = F.interpolate(feature_bchw, size=mask_b1hw.shape[-2:], mode='bilinear')
    batch_pooled_vec = torch.sum(feature_bchw * mask_b1hw, dim = (2, 3)) / (mask_b1hw.sum(dim = (2, 3)) + 1e-5) # B x C
    return torch.mean(batch_pooled_vec, dim=0)

def meta_test_one(cfg, backbone_net, criterion, feature_shape, device, support_img_tensor_bchw, support_mask_tensor_bhw,
                            query_img_tensor_bchw, query_mask_tensor_bhw):
    post_processor = classifier.dispatcher(cfg, feature_shape)
    post_processor = post_processor.to(device)

    num_shots = support_img_tensor_bchw.shape[0]

    # Sanity check to make sure that there are at least some foreground pixels
    for b in range(support_mask_tensor_bhw.shape[0]):
        assert 1 in support_mask_tensor_bhw[b]
    assert 1 in query_mask_tensor_bhw

    # Support set 1. Use masked average pooling to initialize class weight vector to bootstrap fine-tuning
    with torch.no_grad():
        support_feature = backbone_net(support_img_tensor_bchw)
        fg_vec = masked_average_pooling(support_mask_tensor_bhw, support_feature)  # 1 x C
        bg_vec = masked_average_pooling(support_mask_tensor_bhw == 0, support_feature)  # 1 x C
        fg_vec = fg_vec.reshape((1, -1, 1, 1)) # 1xCx1x1
        bg_vec = bg_vec.reshape((1, -1, 1, 1)) # 1xCx1x1
        bg_fg_class_mat = torch.cat([bg_vec, fg_vec], dim=0) #2xCx1x1
        post_processor.pixel_classifier.class_mat.weight.data = bg_fg_class_mat

    # Support set 2. TODO(roger): add back optimization to fine-tune initialized vectors

    # Query set. Evaluation
    with torch.no_grad():
        query_feature = backbone_net(query_img_tensor_bchw)
        eval_ori_spatial_res = query_img_tensor_bchw.shape[-2:]
        eval_predicted_mask = post_processor(query_feature, eval_ori_spatial_res)
        pred_map = eval_predicted_mask.max(dim = 1)[1]

        # TODO(roger): relax this to support multi-way
        # Following PANet, we use ignore_mask to mask confusing pixels from metric
        predicted_fg = torch.logical_and(pred_map == 1, query_mask_tensor_bhw != -1)
        predicted_bg = torch.logical_or(pred_map == 0, query_mask_tensor_bhw == -1)
        tp_cnt = torch.logical_and(predicted_fg, query_mask_tensor_bhw == 1).sum()
        fp_cnt = torch.logical_and(predicted_fg, query_mask_tensor_bhw != 1).sum()
        fn_cnt = torch.logical_and(predicted_bg, query_mask_tensor_bhw == 1).sum()
        tn_cnt = torch.logical_and(predicted_bg, query_mask_tensor_bhw != 1).sum()
    
    return tp_cnt, fp_cnt, fn_cnt, tn_cnt

def construct_meta_test_task(meta_test_set, num_shots):
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
    
    return meta_test_task

def meta_test(cfg, backbone_net, feature_shape, criterion, device, meta_test_set):
    backbone_net.eval()

    # Some assertions and prepare necessary variables
    assert cfg.task.startswith('few_shot')
    assert cfg.DATASET.dataset == "pascal_5i"
    num_shots = cfg.META_TEST.shot
    assert num_shots != -1

    meta_test_task = construct_meta_test_task(meta_test_set, num_shots)

    # Meta Test!
    iou_list = []
    for i, meta_test_single_run in enumerate(meta_test_task):
        for j, meta_test_batch in tqdm(enumerate(meta_test_single_run)):
            novel_class_id = meta_test_batch['novel_class_id']
            query_img, query_target = meta_test_set[meta_test_batch['query_idx']]

            # TODO(roger): modify this to fit multi-way learning
            query_target_mask = (query_target == novel_class_id).long()

            # Incoporate ignore_mask
            query_target_mask[query_target == -1] = -1
            support_img_list = []
            support_mask_list = []
            for supp_idx in meta_test_batch['support_idx_list']:
                supp_img, supp_target = meta_test_set[supp_idx]
                support_img_list.append(supp_img)
                support_mask_list.append(supp_target == novel_class_id)

            # Pad tensor
            query_img_tensor_bchw = query_img.view((1, ) + query_img.shape)
            query_mask_tensor_bhw = (query_target_mask.view((1, ) + query_target_mask.shape)).long()
            support_img_tensor_bchw = torch.stack(support_img_list)
            support_mask_tensor_bhw = torch.stack(support_mask_list).long()
                
            tp_cnt, fp_cnt, fn_cnt, tn_cnt = meta_test_one(cfg, backbone_net, criterion, feature_shape, device,
                            support_img_tensor_bchw.cuda(), support_mask_tensor_bhw.cuda(),
                            query_img_tensor_bchw.cuda(), query_mask_tensor_bhw.cuda())
            
            # Gather episode-wise statistics
            meta_test_task[i][j]['tp_pixel_cnt'] = tp_cnt
            meta_test_task[i][j]['fp_pixel_cnt'] = fp_cnt
            meta_test_task[i][j]['fn_pixel_cnt'] = fn_cnt
            meta_test_task[i][j]['tn_pixel_cnt'] = tn_cnt

        # Gather test numbers
        # TODO(roger): relax this to support dataset other than PASCAL-5i
        total_tp_cnt = np.zeros((5,), dtype = np.int)
        total_fp_cnt = np.zeros((5,), dtype = np.int)
        total_fn_cnt = np.zeros((5,), dtype = np.int)
        total_tn_cnt = np.zeros((5,), dtype = np.int)
        for j, meta_test_batch in enumerate(meta_test_single_run):
            novel_class_id = meta_test_task[i][j]['novel_class_id'] - 1 # from 1-indexed to 0-indexed
            total_tp_cnt[novel_class_id] += meta_test_task[i][j]['tp_pixel_cnt']
            total_fp_cnt[novel_class_id] += meta_test_task[i][j]['fp_pixel_cnt']
            total_fn_cnt[novel_class_id] += meta_test_task[i][j]['fn_pixel_cnt']
            total_tn_cnt[novel_class_id] += meta_test_task[i][j]['tn_pixel_cnt']
        single_run_acc = np.sum(total_tp_cnt + total_tn_cnt) / np.sum(total_tp_cnt + total_fp_cnt + total_fn_cnt + total_tn_cnt)
        classwise_iou = total_tp_cnt / (total_tp_cnt + total_fp_cnt + total_fn_cnt)
        single_run_iou = np.mean(classwise_iou)
        print("Accuracy: {:.4f} IoU: {:.4f}".format(single_run_acc, single_run_iou))
        iou_list.append(single_run_iou)
        
    print("Overall IoU Mean: {:.4f} Std: {:.4f}".format(np.mean(iou_list), np.std(iou_list)))

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
    
    print("Initializing backbone with pretrained weights from: {}".format(args.load))
    trained_weight_dict = torch.load(args.load, map_location=device_str)
    if cfg.BACKBONE.network == "panet_vgg16" and False:
        print("SPECIAL TREATMENT FOR PANET")
        new_dict = {}
        for k in trained_weight_dict:
            new_dict[k[24:]] = trained_weight_dict[k]
        backbone_net.load_state_dict(new_dict, strict=True)
    else:
        backbone_net.load_state_dict(trained_weight_dict['backbone'], strict=True)

    criterion = loss.dispatcher(cfg)

    meta_test(cfg, backbone_net, feature_shape, criterion, device, meta_test_set)

if __name__ == '__main__':
    main()
