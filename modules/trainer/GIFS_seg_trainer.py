import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm, trange

from backbone.deeplabv3_renorm import BatchRenorm2d

import classifier
import utils

from .trainer_base import trainer_base
from dataset.special_loader import get_fs_seg_loader

def harmonic_mean(base_iou, novel_iou):
    return 2 / (1. / base_iou + 1. / novel_iou)

class GIFS_seg_trainer(trainer_base):
    def __init__(self, cfg, backbone_net, post_processor, criterion, dataset_module, device):
        super(GIFS_seg_trainer, self).__init__(cfg, backbone_net, post_processor, criterion, dataset_module, device)

        self.continual_vanilla_train_set = dataset_module.get_continual_vanilla_train_set(cfg)
        self.continual_aug_train_set = dataset_module.get_continual_aug_train_set(cfg)
        self.continual_test_set = dataset_module.get_continual_test_set(cfg)

        self.continual_test_loader = torch.utils.data.DataLoader(self.continual_test_set, batch_size=cfg.TEST.batch_size, shuffle=False, **self.loader_kwargs)

    def train_one(self, device, optimizer, epoch):
        self.backbone_net.train()
        self.post_processor.train()
        start_cp = time.time()
        train_total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            optimizer.zero_grad() # reset gradient
            data, target = data.to(device), target.to(device)
            feature = self.backbone_net(data)
            ori_spatial_res = data.shape[-2:]
            output = self.post_processor(feature, ori_spatial_res)
            loss = self.criterion(output, target)
            loss.backward()
            train_total_loss += loss.item()
            optimizer.step()
            if batch_idx % self.cfg.TRAIN.log_interval == 0:
                pred_map = output.max(dim = 1)[1]
                batch_acc, _ = utils.compute_pixel_acc(pred_map, target, fg_only=self.cfg.METRIC.SEGMENTATION.fg_only)
                print('Train Epoch: {0} [{1}/{2} ({3:.0f}%)]\tLoss: {4:.6f}\tBatch Pixel Acc: {5:.6f} Epoch Elapsed Time: {6:.1f}'.format(
                    epoch, batch_idx * len(data), len(self.train_set),
                    100. * batch_idx / len(self.train_loader), loss.item(), batch_acc, time.time() - start_cp))
        
        return train_total_loss / len(self.train_loader)

    def val_one(self, device):
        self.backbone_net.eval()
        self.post_processor.eval()
        test_loss = 0
        correct = 0
        pixel_acc_list = []
        iou_list = []
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(device), target.to(device)
                feature = self.backbone_net(data)
                ori_spatial_res = data.shape[-2:]
                output = self.post_processor(feature, ori_spatial_res)
                test_loss += self.criterion(output, target).item()  # sum up batch loss
                pred_map = output.max(dim = 1)[1]
                batch_acc, _ = utils.compute_pixel_acc(pred_map, target, fg_only=self.cfg.METRIC.SEGMENTATION.fg_only)
                pixel_acc_list.append(float(batch_acc))
                for i in range(pred_map.shape[0]):
                    iou = utils.compute_iou(
                        np.array(pred_map[i].cpu()),
                        np.array(target[i].cpu(), dtype=np.int64),
                        self.cfg.meta_training_num_classes,
                        fg_only=self.cfg.METRIC.SEGMENTATION.fg_only
                    )
                    iou_list.append(float(iou))

            test_loss /= len(self.val_set)

            m_iou = np.mean(iou_list)
            print('\nTest set: Average loss: {:.4f}, Mean Pixel Accuracy: {:.4f}, Mean IoU {:.4f}'.format(
                test_loss, np.mean(pixel_acc_list), m_iou))
        return m_iou

    def test_one(self, device, visfreq=99999999999, num_runs=5, ft_method="whole_network"):
        assert ft_method in ["none", "whole_network", "classifier_only"]
        num_shots = 5
        
        # Parse image candidates
        testing_label_candidates = self.train_set.dataset.invisible_labels

        image_candidates = {}
        for l in testing_label_candidates:
            image_candidates[l] = sorted(list(self.continual_vanilla_train_set.dataset.get_class_map(l)))

        # We use $num_runs$ consistent random seeds.
        # For each seed, 1000 support-query pairs are sampled.
        np.random.seed(1221)
        seed_list = np.random.randint(0, 99999, size = (num_runs, ))
        
        # Meta Test!
        run_base_iou_list = []
        run_novel_iou_list = []
        run_total_iou_list = []
        run_harm_iou_list = []
        for i in range(num_runs):
            np.random.seed(seed_list[i])
            random.seed(seed_list[i])
            torch.manual_seed(seed_list[i])
            support_set = {}
            for k in image_candidates.keys():
                selected_idx = np.random.choice(image_candidates[k], size=(num_shots, ), replace=False)
                support_set[k] = list(selected_idx)
        
            classwise_iou = self.continual_test_single_pass(support_set, ft_method, visfreq)
            novel_iou_list = []
            base_iou_list = []
            for i in range(len(classwise_iou)):
                label = i + 1 # 0-indexed
                if label in testing_label_candidates:
                    novel_iou_list.append(classwise_iou[i])
                else:
                    base_iou_list.append(classwise_iou[i])
            base_iou = np.mean(base_iou_list)
            novel_iou = np.mean(novel_iou_list)
            print("Base IoU: {:.4f} Novel IoU: {:.4f} Total IoU: {:.4f}".format(base_iou, novel_iou, np.mean(classwise_iou)))
            run_base_iou_list.append(base_iou)
            run_novel_iou_list.append(novel_iou)
            run_total_iou_list.append(np.mean(classwise_iou))
            run_harm_iou_list.append(harmonic_mean(base_iou, novel_iou))
        print("Results of {} runs with {} shots".format(num_runs, num_shots))
        print("Base IoU Mean: {:.4f} Std: {:.4f}".format(np.mean(run_base_iou_list), np.std(run_base_iou_list)))
        print("Novel IoU Mean: {:.4f} Std: {:.4f}".format(np.mean(run_novel_iou_list), np.std(run_novel_iou_list)))
        print("Harmonic IoU Mean: {:.4f} Std: {:.4f}".format(np.mean(run_harm_iou_list), np.std(run_harm_iou_list)))
        print("Total IoU Mean: {:.4f} Std: {:.4f}".format(np.mean(run_total_iou_list), np.std(run_total_iou_list)))
    
    def continual_test_single_pass(self, support_set, ft_method, visfreq):
        temp_backbone_net = deepcopy(self.backbone_net)
        
        self.backbone_net.eval()
        self.post_processor.eval()
        temp_backbone_net.eval()

        num_classes = 21 # 20 fg + 1 bg in VOC
        assert self.post_processor.pixel_classifier.class_mat.weight.data.shape[0] + len(support_set.keys()) == num_classes

        temp_post_processor = classifier.dispatcher(self.cfg, self.feature_shape, num_classes=num_classes)
        temp_post_processor = temp_post_processor.to(self.device)

        # Aggregate weights
        ori_cnt = 0
        class_weight_vec_list = []
        with torch.no_grad():
            for c in range(num_classes):
                if c in support_set:
                    # Aggregate all candidates in support set
                    image_list = []
                    mask_list = []
                    for n_c in support_set:
                        for idx in support_set[n_c]:
                            supp_img_chw, supp_mask_hw = self.continual_vanilla_train_set[idx]
                            if c in supp_mask_hw:
                                image_list.append(supp_img_chw)
                                mask_list.append(supp_mask_hw)
                    # novel class. Use MAP to initialize weight
                    supp_img_bchw_tensor = torch.stack(image_list).cuda()
                    supp_mask_bhw_tensor = torch.stack(mask_list).cuda()
                    # Sanity check to make sure that there are at least some foreground pixels
                    for b in range(supp_mask_bhw_tensor.shape[0]):
                        assert c in supp_mask_bhw_tensor[b]
                    with torch.no_grad():
                        support_feature = self.backbone_net(supp_img_bchw_tensor)
                        class_weight_vec = masked_average_pooling(supp_mask_bhw_tensor == c, support_feature, True)
                else:
                    # base class. Copy weight from learned HEAD
                    class_weight_vec = self.post_processor.pixel_classifier.class_mat.weight.data[ori_cnt]
                    ori_cnt += 1
                class_weight_vec = class_weight_vec.reshape((-1, 1, 1)) # C x 1 x 1
                class_weight_vec_list.append(class_weight_vec)
        
        classifier_weight = torch.stack(class_weight_vec_list) # num_classes x C x 1 x 1
        temp_post_processor.pixel_classifier.class_mat.weight.data = classifier_weight

        # Optimization over support set to fine-tune initialized vectors
        if ft_method == "classifier_only" or ft_method == "whole_network":
            if ft_method == "classifier_only":
                temp_backbone_net.eval()
                temp_post_processor.train()
                trainable_params = temp_post_processor.parameters()
            elif ft_method == "whole_network":
                temp_backbone_net.train()
                temp_post_processor.train()

                trainable_params = [
                    {"params": temp_backbone_net.parameters()},
                    {"params": temp_post_processor.parameters(), "lr": 1e-3}
                ]

                if True:
                    cnt = 0
                    # Freeze batch norm statistics
                    for module in temp_backbone_net.modules():
                        if isinstance(module, nn.BatchNorm2d) or isinstance(module, BatchRenorm2d):
                            if hasattr(module, 'weight'):
                                module.weight.requires_grad_(False)
                            if hasattr(module, 'bias'):
                                module.bias.requires_grad_(False)
                            module.eval()
                            cnt += 1
                    print("Number of batchnorm module frozen: {}".format(cnt))

            # optimizer = optim.Adadelta(trainable_params, lr = 1e-2, weight_decay = 0)
            optimizer = optim.SGD(trainable_params, lr = 1e-3, momentum = 0.9)
            
            max_iter = 1000
            def polynomial_schedule(epoch):
                # from https://arxiv.org/pdf/2012.01415.pdf
                # return 1
                return (1 - epoch / max_iter)**0.9
            batch_size = 5 # min(10, D_n) in GIFS

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, polynomial_schedule)

            # Aggregate elements in support set
            img_idx_list = []
            for c in support_set:
                img_idx_list += support_set[c]

            supp_set_size = len(img_idx_list)

            l2_criterion = nn.MSELoss()

            with trange(1, max_iter + 1, dynamic_ncols=True) as t:
                for iter_i in t:
                    shuffled_idx = torch.randperm(supp_set_size)[:batch_size]
                    image_list = []
                    mask_list = []
                    for idx in shuffled_idx:
                        # Use augmented examples here.
                        img_chw, mask_hw = self.continual_aug_train_set[img_idx_list[idx]]
                        image_list.append(img_chw)
                        mask_list.append(mask_hw)
                    data_bchw = torch.stack(image_list).cuda()
                    target_bhw = torch.stack(mask_list).cuda()
                    feature = temp_backbone_net(data_bchw)
                    ori_spatial_res = data_bchw.shape[-2:]
                    output = temp_post_processor(feature, ori_spatial_res, scale_factor=10)
                    loss = self.criterion(output, target_bhw)
                    if True:
                        # L2 regularization on feature extractor
                        with torch.no_grad():
                            ori_feature = self.backbone_net(data_bchw)
                        regularization_loss = l2_criterion(feature, ori_feature)
                        regularization_loss = regularization_loss * 0.1 # hyperparameter lambda
                        loss = loss + regularization_loss
                        if False:
                            # L2 regulalrization on base classes
                            with torch.no_grad():
                                ori_logit = self.post_processor(ori_feature, ori_spatial_res, scale_factor=10)
                            clf_loss = l2_criterion(output[:,:16,:,:], ori_logit) * 0.1
                            loss = loss + clf_loss
                    optimizer.zero_grad() # reset gradient
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    t.set_description_str("Loss: {:.4f}".format(loss.item()))

        temp_backbone_net.eval()
        temp_post_processor.eval()
        # Evaluation
        return test(self.cfg, temp_backbone_net, temp_post_processor, self.criterion, self.device, self.continual_test_loader, num_classes, visfreq)

def test(cfg, model, post_processor, criterion, device, test_loader, num_classes, visfreq):
    model.eval()
    post_processor.eval()
    test_loss = 0
    correct = 0
    pixel_acc_list = []
    class_intersection, class_union = (None, None)
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            feature = model(data)
            ori_spatial_res = data.shape[-2:]
            output = post_processor(feature, ori_spatial_res)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred_map = output.max(dim = 1)[1]
            batch_acc, _ = utils.compute_pixel_acc(pred_map, target, fg_only=cfg.METRIC.SEGMENTATION.fg_only)
            pixel_acc_list.append(float(batch_acc))
            for i in range(pred_map.shape[0]):
                pred_np = np.array(pred_map[i].cpu())
                target_np = np.array(target[i].cpu(), dtype=np.int64)
                intersection, union = utils.compute_iu(pred_np, target_np, num_classes)
                if class_intersection is None:
                    class_intersection = intersection
                    class_union = union
                else:
                    class_intersection += intersection
                    class_union += union
                if (idx + 1) % visfreq == 0:
                    gt_label = utils.visualize_segmentation(cfg, data[i], target_np, ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"])
                    predicted_label = utils.visualize_segmentation(cfg, data[i], pred_np, ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"])
                    cv2.imwrite("{}_{}_pred.png".format(idx, i), predicted_label)
                    cv2.imwrite("{}_{}_label.png".format(idx, i), gt_label)
                    # Visualize RGB image as well
                    ori_rgb_np = np.array(data[i].permute((1, 2, 0)).cpu())
                    if 'normalize' in cfg.DATASET.TRANSFORM.TEST.transforms:
                        rgb_mean = cfg.DATASET.TRANSFORM.TEST.TRANSFORMS_DETAILS.NORMALIZE.mean
                        rgb_sd = cfg.DATASET.TRANSFORM.TEST.TRANSFORMS_DETAILS.NORMALIZE.sd
                        ori_rgb_np = (ori_rgb_np * rgb_sd) + rgb_mean
                    assert ori_rgb_np.max() <= 1.1, "Max is {}".format(ori_rgb_np.max())
                    ori_rgb_np[ori_rgb_np >= 1] = 1
                    ori_rgb_np = (ori_rgb_np * 255).astype(np.uint8)
                    # Convert to OpenCV BGR
                    ori_rgb_np = cv2.cvtColor(ori_rgb_np, cv2.COLOR_RGB2BGR)
                    cv2.imwrite("{}_{}_ori.jpg".format(idx, i), ori_rgb_np)

    test_loss /= len(test_loader.dataset)

    print("Intersection:")
    print(class_intersection)
    print("Union:")
    print(class_union)
    class_iou = class_intersection / (class_union + 1e-10)
    return class_iou

def masked_average_pooling(mask_b1hw, feature_bchw, normalization):
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
