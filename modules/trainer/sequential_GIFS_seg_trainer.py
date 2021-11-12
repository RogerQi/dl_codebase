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
        super(sequential_GIFS_seg_trainer, self).__init__(cfg, backbone_net, post_processor, criterion, dataset_module, device)

        self.continual_vanilla_train_set = dataset_module.get_continual_vanilla_train_set(cfg)
        self.continual_aug_train_set = dataset_module.get_continual_aug_train_set(cfg)
        self.continual_test_set_list = dataset_module.get_sequential_continual_test_set(cfg)
        
        self.test_loader_list = []

        for s in self.continual_test_set_list:
            self.test_loader_list.append(torch.utils.data.DataLoader(s, batch_size=cfg.TEST.batch_size, shuffle=False, **self.loader_kwargs))
        
        self.partial_data_pool = {}
    
    def continual_test_single_pass(self, support_set):
        self.partial_data_pool = {}
        self.vanilla_backbone_net = deepcopy(self.backbone_net)
        self.vanilla_post_processor = deepcopy(self.post_processor)

        all_novel_class_idx = sorted(list(support_set.keys()))
        base_class_idx = self.train_set.dataset.visible_labels
        if 0 not in base_class_idx:
            base_class_idx.append(0)
        base_class_idx = sorted(base_class_idx)

        self.vanilla_base_class_idx = deepcopy(base_class_idx)
        learned_novel_class_idx = []

        total_num_classes = len(all_novel_class_idx) + len(base_class_idx)

        # Construct task batches
        assert len(all_novel_class_idx) % self.cfg.TASK_SPECIFIC.GIFS.sequential_dataset_num_classes == 0
        num_tasks = len(all_novel_class_idx) // self.cfg.TASK_SPECIFIC.GIFS.sequential_dataset_num_classes
        ptr = 0
        task_stream = []
        for i in range(num_tasks):
            current_task = []
            for j in range(self.cfg.TASK_SPECIFIC.GIFS.sequential_dataset_num_classes):
                current_task.append(all_novel_class_idx[ptr])
                ptr += 1
            task_stream.append(current_task)
        assert ptr == len(all_novel_class_idx)

        for i, task in enumerate(task_stream):
            # Aggregate elements in support set
            image_list = []
            mask_list = []

            for n_c in task:
                for idx in support_set[n_c]:
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

            if len(base_class_idx) != self.prv_post_processor.pixel_classifier.class_mat.weight.data.shape[0]:
                # squeeze the classifier weights
                print("Base IDX list and pixel classifier weight mismatched! {}".format(base_class_idx))
                self.prv_post_processor.pixel_classifier.class_mat.weight.data = self.prv_post_processor.pixel_classifier.class_mat.weight.data[base_class_idx]

            self.novel_adapt(base_class_idx, task, supp_img_bchw, supp_mask_bhw)
            learned_novel_class_idx += task

            # Evaluation
            if i == len(task_stream) - 1:
                classwise_iou = self.eval_on_loader(self.test_loader_list[i], total_num_classes)

                classwise_iou = np.array(classwise_iou)

                # to handle background and 0-indexing
                novel_iou_list = []
                base_iou_list = []
                for i in range(len(classwise_iou)):
                    label = i + 1 # 0-indexed
                    if label in learned_novel_class_idx:
                        novel_iou_list.append(classwise_iou[i])
                    elif label in self.vanilla_base_class_idx:
                        base_iou_list.append(classwise_iou[i])
                    else:
                        continue
                base_iou = np.mean(base_iou_list)
                novel_iou = np.mean(novel_iou_list)

                print("Base IoU: {:.4f} Novel IoU: {:.4f}".format(base_iou, novel_iou))
                print("Novel class wise IoU: {}".format(novel_iou_list))

            base_class_idx += task
            base_class_idx = sorted(base_class_idx)

        # Restore weights
        self.backbone_net = self.vanilla_backbone_net
        self.post_processor = self.vanilla_post_processor

        return classwise_iou
