from modules.trainer.GIFS_seg_trainer import GIFS_seg_trainer
import os
import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
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

from .GIFS_seg_trainer import GIFS_seg_trainer
from dataset.special_loader import get_fs_seg_loader

from IPython import embed

def harmonic_mean(base_iou, novel_iou):
    return 2 / (1. / base_iou + 1. / novel_iou)

class sequential_GIFS_seg_trainer(GIFS_seg_trainer):
    def __init__(self, cfg, backbone_net, post_processor, criterion, dataset_module, device):
        super(GIFS_seg_trainer, self).__init__(cfg, backbone_net, post_processor, criterion, dataset_module, device)

        self.continual_vanilla_train_set = dataset_module.get_continual_vanilla_train_set(cfg)
        self.continual_aug_train_set = dataset_module.get_continual_aug_train_set(cfg)
        self.continual_test_set_list = dataset_module.get_sequential_continual_test_set(cfg)
        
        self.test_loader_list = []

        for s in self.continual_test_set_list:
            self.test_loader_list.append(torch.utils.data.DataLoader(s, batch_size=cfg.TEST.batch_size, shuffle=False, **self.loader_kwargs))
    
    def continual_test_single_pass(self, support_set):
        vanilla_backbone_net = deepcopy(self.backbone_net)
        vanilla_post_processor = deepcopy(self.post_processor)

        all_novel_class_idx = sorted(list(support_set.keys()))
        base_class_idx = self.train_set.dataset.visible_labels
        if 0 not in base_class_idx:
            base_class_idx.append(0)
        base_class_idx = sorted(base_class_idx)

        total_num_classes = len(all_novel_class_idx) + len(base_class_idx)

        for i, c in enumerate(all_novel_class_idx):
            novel_class_idx = [c]
            # Aggregate elements in support set
            image_list = []
            mask_list = []

            for idx in support_set[c]:
                img_chw, mask_hw = self.continual_vanilla_train_set[idx]
                image_list.append(img_chw)
                mask_list.append(mask_hw)

            supp_img_bchw = torch.stack(image_list)
            supp_mask_bhw = torch.stack(mask_list)

            self.prv_backbone_net = deepcopy(self.backbone_net)
            self.prv_post_processor = deepcopy(self.post_processor)
            
            self.prv_backbone_net.eval()
            self.prv_post_processor.eval()
            self.backbone_net.eval()
            self.post_processor.eval()

            assert len(base_class_idx) == self.prv_post_processor.pixel_classifier.class_mat.weight.data.shape[0]

            self.novel_adapt(base_class_idx, novel_class_idx, supp_img_bchw, supp_mask_bhw)

            # Evaluation
            classwise_iou = self.eval_on_loader(self.test_loader_list[i], total_num_classes)

            classwise_iou = np.array(classwise_iou)

            # to handle background and 0-indexing
            novel_iou_list = []
            base_iou_list = []
            for i in range(len(classwise_iou)):
                label = i + 1 # 0-indexed
                if label in novel_class_idx:
                    novel_iou_list.append(classwise_iou[i])
                elif label in base_class_idx:
                    base_iou_list.append(classwise_iou[i])
                else:
                    continue
            base_iou = np.mean(base_iou_list)
            novel_iou = np.mean(novel_iou_list)

            print("Base IoU: {:.4f} Class IoU: {:.4f}".format(base_iou, novel_iou))

            base_class_idx.append(c)
            base_class_idx = sorted(base_class_idx)

        # Restore weights
        self.backbone_net = vanilla_backbone_net
        self.post_processor = vanilla_post_processor

        return classwise_iou
