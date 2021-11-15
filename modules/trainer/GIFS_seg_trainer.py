import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import List, Union

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as tr_F
from tqdm import tqdm, trange

from backbone.deeplabv3_renorm import BatchRenorm2d

import classifier
import utils

from .seg_trainer import seg_trainer

from IPython import embed

def harmonic_mean(base_iou, novel_iou):
    return 2 / (1. / base_iou + 1. / novel_iou)

class GIFS_seg_trainer(seg_trainer):
    def __init__(self, cfg, backbone_net, post_processor, criterion, dataset_module, device):
        super(GIFS_seg_trainer, self).__init__(cfg, backbone_net, post_processor, criterion, dataset_module, device)

        self.continual_vanilla_train_set = dataset_module.get_continual_vanilla_train_set(cfg)
        self.continual_aug_train_set = dataset_module.get_continual_aug_train_set(cfg)
        self.continual_test_set = dataset_module.get_continual_test_set(cfg)

        self.continual_test_loader = torch.utils.data.DataLoader(self.continual_test_set, batch_size=cfg.TEST.batch_size, shuffle=False, **self.loader_kwargs)

    # self.train_one is inherited from seg trainer
    # self.val_one is inherited from seg trainer

    def test_one(self, device, num_runs=5):
        num_shots = self.cfg.TASK_SPECIFIC.GIFS.num_shots
        
        # Parse image candidates
        testing_label_candidates = self.train_set.dataset.invisible_labels

        vanilla_image_candidates = {}
        for l in testing_label_candidates:
            vanilla_image_candidates[l] = set(self.continual_vanilla_train_set.dataset.get_class_map(l))
        
        # To ensure only $num_shots$ number of examples are used
        image_candidates = {}
        for k_i in vanilla_image_candidates:
            image_candidates[k_i] = deepcopy(vanilla_image_candidates[k_i])
            for k_j in vanilla_image_candidates:
                if k_i == k_j: continue
                image_candidates[k_i] -= vanilla_image_candidates[k_j]
            image_candidates[k_i] = sorted(list(image_candidates[k_i]))

        # We use a total of $num_runs$ consistent random seeds.
        np.random.seed(1234)
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
        
            # get per-class IoU on the entire validation set based on results from the support set
            classwise_iou = self.continual_test_single_pass(support_set)

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
    
    def classifier_weight_imprinting(self, base_id_list: List[int], novel_id_list: List[int], supp_img_bchw: torch.Tensor, supp_mask_bhw: torch.Tensor):
        """Use masked average pooling to initialize a new 1x1 convolutional HEAD for semantic segmentation

        The resulting classifier will produce per-pixel classification from class 0 (usually background)
        upto class max(max(base_class_idx), max(novel_class_idx)). If there is discontinuity in base_class_idx
        and novel_class_idx (e.g., base: [0, 1, 2, 4]; novel: [5, 6]), then the class weight of the non-used class
        will be initialized as full zeros.

        Args:
            base_id_list (List[int]): a sorted list containing base class id
            novel_id_list (List[int]): a sorted list containing novel class id
            supp_img_bchw (torch.Tensor): Normalized support set image tensor
            supp_mask_bhw (torch.Tensor): Complete segmentation mask of support set

        Returns:
            torch.Tensor: a weight vector that can be directly plugged back to
                data.weight of the 1x1 classification convolution
        """
        assert self.prv_backbone_net is not None
        assert self.prv_post_processor is not None
        max_cls = max(max(base_id_list), max(novel_id_list)) + 1
        assert self.prv_post_processor.pixel_classifier.class_mat.weight.data.shape[0] == len(base_id_list)

        ori_cnt = 0
        class_weight_vec_list = []
        for c in range(max_cls):
            if c in novel_id_list:
                # Aggregate all candidates in support set
                image_list = []
                mask_list = []
                for b in range(supp_img_bchw.shape[0]):
                    if c in supp_mask_bhw[b]:
                        image_list.append(supp_img_bchw[b])
                        mask_list.append(supp_mask_bhw[b])
                assert image_list, "no novel example found"
                assert mask_list, "no novel example found"
                # novel class. Use MAP to initialize weight
                supp_img_bchw_tensor = torch.stack(image_list).cuda()
                supp_mask_bhw_tensor = torch.stack(mask_list).cuda()
                with torch.no_grad():
                    support_feature = self.prv_backbone_net(supp_img_bchw_tensor)
                    class_weight_vec = utils.masked_average_pooling(supp_mask_bhw_tensor == c, support_feature, True)
            elif c in base_id_list:
                # base class. Copy weight from learned HEAD
                class_weight_vec = self.prv_post_processor.pixel_classifier.class_mat.weight.data[ori_cnt]
                ori_cnt += 1
            else:
                # not used class
                class_weight_vec = torch.zeros_like(self.prv_post_processor.pixel_classifier.class_mat.weight.data[0])
            class_weight_vec = class_weight_vec.reshape((-1, 1, 1)) # C x 1 x 1
            class_weight_vec_list.append(class_weight_vec)
        
        classifier_weights = torch.stack(class_weight_vec_list) # num_classes x C x 1 x 1
        return classifier_weights
    
    def finetune_backbone(self, base_class_idx, novel_class_idx, supp_img_bchw, supp_mask_bhw):
        assert self.prv_backbone_net is not None
        assert self.prv_post_processor is not None

        self.backbone_net.train()
        self.post_processor.train()

        trainable_params = [
            {"params": self.backbone_net.parameters()},
            {"params": self.post_processor.parameters(), "lr": self.cfg.TASK_SPECIFIC.GIFS.classifier_lr}
        ]

        # Freeze batch norm statistics
        for module in self.backbone_net.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, BatchRenorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()

        optimizer = optim.SGD(trainable_params, lr = self.cfg.TASK_SPECIFIC.GIFS.backbone_lr, momentum = 0.9)
        
        max_iter = self.cfg.TASK_SPECIFIC.GIFS.max_iter
        def polynomial_schedule(epoch):
            return (1 - epoch / max_iter)**0.9
        batch_size = self.cfg.TASK_SPECIFIC.GIFS.ft_batch_size

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, polynomial_schedule)

        l2_criterion = nn.MSELoss()

        with trange(1, max_iter + 1, dynamic_ncols=True) as t:
            for iter_i in t:
                shuffled_idx = torch.randperm(supp_img_bchw.shape[0])[:batch_size]
                image_list = []
                mask_list = []
                for idx in shuffled_idx:
                    # Use augmented examples here.
                    img_chw = supp_img_bchw[idx]
                    mask_hw = supp_mask_bhw[idx]
                    if torch.rand(1) < 0.5:
                        img_chw = tr_F.hflip(img_chw)
                        mask_hw = tr_F.hflip(mask_hw)
                    image_list.append(img_chw)
                    mask_list.append(mask_hw)
                data_bchw = torch.stack(image_list).cuda()
                target_bhw = torch.stack(mask_list).cuda()
                feature = self.backbone_net(data_bchw)
                ori_spatial_res = data_bchw.shape[-2:]
                output = self.post_processor(feature, ori_spatial_res, scale_factor=10)

                # L2 regularization on feature extractor
                with torch.no_grad():
                    ori_feature = self.prv_backbone_net(data_bchw)
                    ori_logit = self.prv_post_processor(ori_feature, ori_spatial_res, scale_factor=10)

                if self.cfg.TASK_SPECIFIC.GIFS.pseudo_base_label:
                    novel_mask = torch.zeros_like(target_bhw)
                    for novel_idx in novel_class_idx:
                        novel_mask = torch.logical_or(novel_mask, target_bhw == novel_idx)
                    tmp_target_bhw = output.max(dim = 1)[1]
                    tmp_target_bhw[novel_mask] = target_bhw[novel_mask]
                    target_bhw = tmp_target_bhw

                loss = self.criterion(output, target_bhw)

                # Feature extractor regularization + classifier regularization
                regularization_loss = l2_criterion(feature, ori_feature)
                regularization_loss = regularization_loss * self.cfg.TASK_SPECIFIC.GIFS.feature_reg_lambda # hyperparameter lambda
                loss = loss + regularization_loss
                # L2 regulalrization on base classes
                clf_loss = l2_criterion(output[:,base_class_idx,:,:], ori_logit) * self.cfg.TASK_SPECIFIC.GIFS.classifier_reg_lambda
                loss = loss + clf_loss

                optimizer.zero_grad() # reset gradient
                loss.backward()
                optimizer.step()
                scheduler.step()
                t.set_description_str("Loss: {:.4f}".format(loss.item()))
    
    def continual_test_single_pass(self, support_set):
        self.prv_backbone_net = deepcopy(self.backbone_net)
        self.prv_post_processor = deepcopy(self.post_processor)
        
        self.prv_backbone_net.eval()
        self.prv_post_processor.eval()
        self.backbone_net.eval()
        self.post_processor.eval()

        n_base_classes = self.prv_post_processor.pixel_classifier.class_mat.weight.data.shape[0]
        n_novel_classes = len(support_set.keys())
        num_classes = n_base_classes + n_novel_classes

        novel_class_idx = sorted(list(support_set.keys()))
        base_class_idx = [i for i in range(num_classes) if i not in novel_class_idx]

        # Aggregate elements in support set
        image_list = []
        mask_list = []

        for c in support_set:
            for idx in support_set[c]:
                img_chw, mask_hw = self.continual_vanilla_train_set[idx]
                image_list.append(img_chw)
                mask_list.append(mask_hw)
        supp_img_bchw = torch.stack(image_list)
        supp_mask_bhw = torch.stack(mask_list)

        self.novel_adapt(base_class_idx, novel_class_idx, supp_img_bchw, supp_mask_bhw)

        # Evaluation
        metric = self.eval_on_loader(self.continual_test_loader, num_classes)

        # Restore weights
        self.backbone_net = self.prv_backbone_net
        self.post_processor = self.prv_post_processor

        return metric
    
    def novel_adapt(self, base_class_idx, novel_class_idx, supp_img_bchw, supp_mask_bhw):
        max_cls = max(max(base_class_idx), max(novel_class_idx)) + 1
        self.post_processor = classifier.dispatcher(self.cfg, self.feature_shape, num_classes=max_cls)
        self.post_processor = self.post_processor.to(self.device)
        # Aggregate weights
        aggregated_weights = self.classifier_weight_imprinting(base_class_idx, novel_class_idx, supp_img_bchw, supp_mask_bhw)
        self.post_processor.pixel_classifier.class_mat.weight.data = aggregated_weights

        # Optimization over support set to fine-tune initialized vectors
        if self.cfg.TASK_SPECIFIC.GIFS.fine_tuning:
            self.finetune_backbone(base_class_idx, novel_class_idx, supp_img_bchw, supp_mask_bhw)
