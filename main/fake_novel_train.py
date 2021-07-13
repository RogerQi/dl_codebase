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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from IPython import embed

def parse_args():
    parser = argparse.ArgumentParser(description = "Roger's Deep Learning Playground")
    parser.add_argument('--cfg', help = "specify particular yaml configuration to use", required = True,
        default = "configs/mnist_torch_official.yaml", type = str)
    parser.add_argument('--resume', help = "resume training from checkpoint", required=False, default = "NA", type = str)
    args = parser.parse_args()

    return args

def masked_average_pooling(mask_b1hw, feature_bchw, normalization=False):
    '''
    Params
        - mask_b1hw: a binary mask whose element-wise value is either 0 or 1
        - feature_bchw: feature map obtained from the backbone
    
    Return: Mask-average-pooled vector of shape 1 x C
    '''
    if len(mask_b1hw.shape) == 3:
        mask_b1hw = mask_b1hw.view((mask_b1hw.shape[0], 1, mask_b1hw.shape[1], mask_b1hw.shape[2]))

    # Assert remove mask is not in mask provided
    assert -1 not in mask_b1hw

    # Spatial resolution mismatched. Interpolate feature to match mask size
    if mask_b1hw.shape[-2:] != feature_bchw.shape[-2:]:
        feature_bchw = F.interpolate(feature_bchw, size=mask_b1hw.shape[-2:], mode='bilinear')
    
    if normalization:
        feature_norm = torch.norm(feature_bchw, p=2, dim=1).unsqueeze(1).expand_as(feature_bchw)
        feature_bchw = feature_bchw.div(feature_norm + 1e-5) # avoid div by zero

    batch_pooled_vec = torch.sum(feature_bchw * mask_b1hw, dim = (2, 3)) / (mask_b1hw.sum(dim = (2, 3)) + 1e-5) # B x C
    return torch.mean(batch_pooled_vec, dim=0)

def train(cfg, model, post_processor, criterion, device, train_loader, optimizer, epoch, train_set):
    model.train()
    post_processor.train()
    start_cp = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad() # reset gradient
        if torch.rand(1) < 0.05:
            fake_novel_flag = True
        else:
            fake_novel_flag = False
        if fake_novel_flag:
            with torch.no_grad():
                selected_class_idx = np.random.randint(low=0, high=15)
                image_candidates = train_set.dataset.get_class_map(selected_class_idx + 1) # 0-indexed to 1-indexed
                selected_images = np.random.choice(image_candidates, size=(5,), replace=False) # 5 shot
                supp_image_bchw, supp_mask_bhw = zip(*[train_set[i] for i in selected_images])
                supp_image_bchw = torch.stack(supp_image_bchw).cuda()
                supp_mask_bhw = torch.stack(supp_mask_bhw).cuda()
                supp_feature = model(supp_image_bchw)
                for b in range(supp_mask_bhw.shape[0]):
                    assert selected_class_idx + 1 in supp_mask_bhw[b]
                fake_novel_vec = masked_average_pooling(supp_mask_bhw == (selected_class_idx + 1), supp_feature, True)
                fake_novel_vec = fake_novel_vec.view((-1, 1, 1)) # C x 1 x 1
                original_vec = post_processor.pixel_classifier.class_mat.weight.data[selected_class_idx]
                post_processor.pixel_classifier.class_mat.weight.data[selected_class_idx] = fake_novel_vec
        data, target = data.to(device), target.to(device)
        feature = model(data)
        # Generate fake novel vector
        assert cfg.task != "classification"
        ori_spatial_res = data.shape[-2:]
        output = post_processor(feature, ori_spatial_res)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if fake_novel_flag:
            with torch.no_grad():
                # Restore original class weight after weight update
                post_processor.pixel_classifier.class_mat.weight.data[selected_class_idx] = original_vec
        if batch_idx % cfg.TRAIN.log_interval == 0:
            pred_map = output.max(dim = 1)[1]
            batch_acc, _ = utils.compute_pixel_acc(pred_map, target, fg_only=cfg.METRIC.SEGMENTATION.fg_only)
            print('Train Epoch: {0} [{1}/{2} ({3:.0f}%)]\tLoss: {4:.6f}\tBatch Pixel Acc: {5:.6f} Epoch Elapsed Time: {6:.1f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), batch_acc, time.time() - start_cp))

def test(cfg, model, post_processor, criterion, device, test_loader):
    """
    Return: a validation metric between 0-1 where 1 is perfect
    """
    model.eval()
    post_processor.eval()
    test_loss = 0
    correct = 0
    # TODO: use a more consistent evaluation interface
    pixel_acc_list = []
    iou_list = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            feature = model(data)
            if cfg.task == "classification":
                output = post_processor(feature)
            elif cfg.task == "semantic_segmentation" or cfg.task == "few_shot_semantic_segmentation_fine_tuning":
                ori_spatial_res = data.shape[-2:]
                output = post_processor(feature, ori_spatial_res)
            test_loss += criterion(output, target).item()  # sum up batch loss
            if cfg.task == "classification":
                pred = output.argmax(dim = 1, keepdim = True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            elif cfg.task == "semantic_segmentation":
                pred_map = output.max(dim = 1)[1]
                batch_acc, _ = utils.compute_pixel_acc(pred_map, target, fg_only=cfg.METRIC.SEGMENTATION.fg_only)
                pixel_acc_list.append(float(batch_acc))
                for i in range(pred_map.shape[0]):
                    iou = utils.compute_iou(
                        np.array(pred_map[i].cpu()),
                        np.array(target[i].cpu(), dtype=np.int64),
                        cfg.num_classes,
                        fg_only=cfg.METRIC.SEGMENTATION.fg_only
                    )
                    iou_list.append(float(iou))
            elif cfg.task == "few_shot_semantic_segmentation_fine_tuning":
                pred_map = output.max(dim = 1)[1]
                batch_acc, _ = utils.compute_pixel_acc(pred_map, target, fg_only=cfg.METRIC.SEGMENTATION.fg_only)
                pixel_acc_list.append(float(batch_acc))
                for i in range(pred_map.shape[0]):
                    iou = utils.compute_iou(
                        np.array(pred_map[i].cpu()),
                        np.array(target[i].cpu(), dtype=np.int64),
                        cfg.meta_training_num_classes,
                        fg_only=cfg.METRIC.SEGMENTATION.fg_only
                    )
                    iou_list.append(float(iou))
            else:
                raise NotImplementedError

        test_loss /= len(test_loader.dataset)

        if cfg.task == "classification":
            acc = 100. * correct / len(test_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, len(test_loader.dataset), acc))
            return acc
        elif cfg.task == "semantic_segmentation" or cfg.task == "few_shot_semantic_segmentation_fine_tuning":
            m_iou = np.mean(iou_list)
            print('\nTest set: Average loss: {:.4f}, Mean Pixel Accuracy: {:.4f}, Mean IoU {:.4f}'.format(
                test_loss, np.mean(pixel_acc_list), m_iou))
            return m_iou
        else:
            raise NotImplementedError

def main():
    # --------------------------
    # | Initial set up
    # |  1. Parse arg and yaml
    # |  2. Set device
    # |  3. Set seed
    # --------------------------
    args = parse_args()
    update_config_from_yaml(cfg, args)

    print(cfg)

    use_cuda = not cfg.SYSTEM.use_cpu
    device_str = "cuda" if use_cuda else "cpu"
    device = torch.device(device_str)

    kwargs = {'num_workers': cfg.SYSTEM.num_workers, 'pin_memory': cfg.SYSTEM.pin_memory} if use_cuda else {}

    torch.manual_seed(cfg.seed)

    # --------------------------
    # | Prepare datasets
    # --------------------------
    train_set, test_set = dataset.dispatcher(cfg)

    print("Training set contains {} data points.".format(len(train_set)))
    print("Test/Val set contains {} data points.".format(len(test_set)))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.TRAIN.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.TEST.batch_size, shuffle=True, **kwargs)

    # --------------------------
    # | Get ready to learn
    # |  1. Prepare network and loss
    # |  2. Prepare optimizer
    # |  3. Set learning rate
    # --------------------------
    backbone_net = backbone.dispatcher(cfg)
    backbone_net = backbone_net(cfg).to(device)
    feature_shape = backbone_net.get_feature_tensor_shape(device)
    print("Backbone output feature tensor shape: {}".format(feature_shape))
    post_processor = classifier.dispatcher(cfg, feature_shape)
    
    post_processor = post_processor.to(device)
    
    if cfg.BACKBONE.use_pretrained:
        weight_path = cfg.BACKBONE.pretrained_path
        print("Initializing backbone with pretrained weights from: {}".format(weight_path))
        pretrained_weight_dict = torch.load(weight_path, map_location=device_str)
        if cfg.BACKBONE.network == 'panet_vgg16':
            keys = list(pretrained_weight_dict.keys())
            new_dict = backbone_net.state_dict()
            new_keys = list(new_dict.keys())

            for i in range(26):
                new_dict[new_keys[i]] = pretrained_weight_dict[keys[i]]
            
            backbone_net.load_state_dict(new_dict, strict=True)
        else:
            backbone_net.load_state_dict(pretrained_weight_dict, strict=False)

    start_epoch = 1
    if args.resume != "NA":
        sub_str = args.resume[args.resume.index('epoch') + 5:]
        start_epoch = int(sub_str[:sub_str.index('_')]) + 1
        assert start_epoch < cfg.TRAIN.max_epochs
        print("Resuming training from epoch {}".format(start_epoch))
        trained_weight_dict = torch.load(args.resume, map_location=device_str)
        backbone_net.load_state_dict(trained_weight_dict['backbone'], strict=True)
        post_processor.load_state_dict(trained_weight_dict['head'], strict=True)

    criterion = loss.dispatcher(cfg)

    trainable_params = list(backbone_net.parameters()) + list(post_processor.parameters())

    if cfg.TRAIN.OPTIMIZER.type == "adadelta":
        optimizer = optim.Adadelta(trainable_params, lr = cfg.TRAIN.initial_lr,
                                    weight_decay = cfg.TRAIN.OPTIMIZER.weight_decay)
    elif cfg.TRAIN.OPTIMIZER.type == "SGD":
        optimizer = optim.SGD(trainable_params, lr = cfg.TRAIN.initial_lr, momentum = cfg.TRAIN.OPTIMIZER.momentum,
                                weight_decay = cfg.TRAIN.OPTIMIZER.weight_decay)
    elif cfg.TRAIN.OPTIMIZER.type == "ADAM":
        optimizer = optim.Adam(trainable_params, lr = cfg.TRAIN.initial_lr, betas = (0.9, 0.999),
                                weight_decay = cfg.TRAIN.OPTIMIZER.weight_decay)
    else:
        raise NotImplementedError("Got unsupported optimizer: {}".format(cfg.TRAIN.OPTIMIZER.type))

    # Prepare LR scheduler
    if cfg.TRAIN.lr_scheduler == "step_down":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = cfg.TRAIN.step_down_on_epoch,
                                                            gamma = cfg.TRAIN.step_down_gamma)
    elif cfg.TRAIN.lr_scheduler == "polynomial":
        max_epoch = cfg.TRAIN.max_epochs
        def polynomial_schedule(epoch):
            # from https://arxiv.org/pdf/2012.01415.pdf
            return (1 - epoch / max_epoch)**0.9
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, polynomial_schedule)
    else:
        raise NotImplementedError("Got unsupported scheduler: {}".format(cfg.TRAIN.lr_scheduler))

    best_val_metric = 0

    # Tune LR scheduler
    for epoch in range(1, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, cfg.TRAIN.max_epochs + 1):
        start_cp = time.time()
        train(cfg, backbone_net, post_processor, criterion, device, train_loader, optimizer, epoch, train_set)
        scheduler.step()
        print("Training took {:.4f} seconds".format(time.time() - start_cp))
        start_cp = time.time()
        val_metric = test(cfg, backbone_net, post_processor, criterion, device, test_loader)
        print("Eval took {:.4f} seconds.".format(time.time() - start_cp))
        if val_metric > best_val_metric:
            print("Epoch {} New Best Model w/ metric: {:.4f}".format(epoch, val_metric))
            best_val_metric = val_metric
            if cfg.save_model:
                best_model_path = "{0}_epoch{1}_{2:.4f}.pt".format(cfg.name, epoch, best_val_metric)
                print("Saving model to {}".format(best_model_path))
                torch.save(
                    {
                        "backbone": backbone_net.state_dict(),
                        "head": post_processor.state_dict()
                    },
                    best_model_path
                )
        print("===================================\n")

    if cfg.save_model:
        torch.save(
                {
                    "backbone": backbone_net.state_dict(),
                    "head": post_processor.state_dict()
                },
                "{0}_final.pt".format(cfg.name)
            )


if __name__ == '__main__':
    main()
