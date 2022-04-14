import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from copy import deepcopy
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms.functional as tr_F
from tqdm import tqdm, trange
from backbone.deeplabv3_renorm import BatchRenorm2d

import utils

from .seg_trainer import seg_trainer
from IPython import embed

memory_bank_size = 500

class live_continual_seg_trainer(seg_trainer):
    def __init__(self, cfg, backbone_net, post_processor, criterion, dataset_module, device):
        super(live_continual_seg_trainer, self).__init__(cfg, backbone_net, post_processor, criterion, dataset_module, device)
        
        self.psuedo_database = {}
        self.img_name_id_map = {}
    
    def test_one(self, device):
        self.backbone_net.eval()
        self.post_processor.eval()
        self.base_img_candidates = self.construct_baseset()
        normalizer = tv.transforms.Normalize(mean=self.cfg.DATASET.TRANSFORM.TRAIN.TRANSFORMS_DETAILS.NORMALIZE.mean,
                                    std=self.cfg.DATASET.TRANSFORM.TRAIN.TRANSFORMS_DETAILS.NORMALIZE.sd)
        # load test video
        novel_idx = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78]
        our_class_name = [self.continual_test_set.dataset.CLASS_NAMES_LIST[i] for i in range(81) if i not in novel_idx]
        suitcase_idx = our_class_name.index('suitcase')
        self.suitcase_idx = suitcase_idx
        self.post_processor.pixel_classifier.class_mat.weight.data[suitcase_idx] = torch.zeros_like(self.post_processor.pixel_classifier.class_mat.weight.data[suitcase_idx])
        our_class_name.append('suitcase')
        video_path = 'demo_airport.mp4'
        cap = cv2.VideoCapture(video_path)
        out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 30.0, (640,360))
        cnt = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            cv2.imshow('raw', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            rgb_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_chw = torch.tensor(rgb_np).float().permute((2, 0, 1))
            img_chw = img_chw / 255 # norm to 0-1
            img_chw = normalizer(img_chw)
            img_bchw = img_chw.view((1,) + img_chw.shape)
            pred_map = self.infer_one(img_bchw).cpu().numpy()[0]
            label_vis = utils.visualize_segmentation(self.cfg, img_chw, pred_map, our_class_name)
            cv2.imshow('label', cv2.cvtColor(label_vis, cv2.COLOR_RGB2BGR))
            out.write(cv2.cvtColor(label_vis, cv2.COLOR_RGB2BGR))
            if cnt == 25:
                # cv2.imwrite("1.jpg", frame)
                mask = Image.open('1_mask.png')
                mask = np.array(mask)
                mask = mask / 255
                mask = torch.tensor(mask).int()
                self.novel_adapt_single(img_chw, mask, 'suitcase')
            if cnt == 210:
                # cv2.imwrite("2.jpg", frame)
                mask = Image.open('2_mask.png')
                mask = np.array(mask)
                mask = mask / 255
                mask = torch.tensor(mask).int()
                self.novel_adapt_single(img_chw, mask, 'suitcase')
            if cnt == 240:
                # cv2.imwrite("3.jpg", frame)
                mask = Image.open('3_mask.png')
                mask = np.array(mask)
                mask = mask / 255
                mask = torch.tensor(mask).int()
                self.novel_adapt_single(img_chw, mask, 'suitcase')
            cnt += 1
        out.release()
    
    def infer_one(self, img_bchw):
       # Inference
        data = img_bchw.to(self.device)
        feature = self.backbone_net(data)
        ori_spatial_res = data.shape[-2:]
        output = self.post_processor(feature, ori_spatial_res)
        pred_map = output.max(dim = 1)[1]
        return pred_map
    
    def novel_adapt_single(self, img_chw, mask_hw, obj_name):
        """Adapt to a single image

        Args:
            img (torch.Tensor): Normalized RGB image tensor of shape (3, H, W)
            mask (torch.Tensor): Binary mask of novel object
        """
        num_existing_class = self.post_processor.pixel_classifier.class_mat.weight.data.shape[0]
        img_roi, mask_roi = utils.crop_partial_img(img_chw, mask_hw)
        if obj_name not in self.psuedo_database:
            self.psuedo_database[obj_name] = [(img_chw, mask_hw, img_roi, mask_roi)]
            new_clf_weights = self.classifier_weight_imprinting_one(img_chw, mask_hw)
            self.post_processor.pixel_classifier.class_mat.weight.data = new_clf_weights
            self.img_name_id_map[obj_name] = num_existing_class # 0-indexed shift 1
        else:
            self.psuedo_database[obj_name].append((img_chw, mask_hw, img_roi, mask_roi))
        self.finetune_backbone_one(obj_name)

    def classifier_weight_imprinting_one(self, supp_img_chw, supp_mask_hw):
        """Use masked average pooling to initialize a new 1x1 convolutional HEAD for semantic segmentation

        The resulting classifier will produce per-pixel classification from class 0 (usually background)
        upto class max(max(base_class_idx), max(novel_class_idx)). If there is discontinuity in base_class_idx
        and novel_class_idx (e.g., base: [0, 1, 2, 4]; novel: [5, 6]), then the class weight of the non-used class
        will be initialized as full zeros.

        Args:
            supp_img_chw (torch.Tensor): Normalized support set image tensor
            supp_mask_hw (torch.Tensor): Complete segmentation mask of support set

        Returns:
            torch.Tensor: a weight vector that can be directly plugged back to
                data.weight of the 1x1 classification convolution
        """
        class_weight_vec_list = [self.post_processor.pixel_classifier.class_mat.weight.data]
        # novel class. Use MAP to initialize weight
        supp_img_bchw_tensor = supp_img_chw.reshape((1,) + supp_img_chw.shape).to(self.device)
        supp_mask_bhw_tensor = supp_mask_hw.reshape((1,) + supp_mask_hw.shape).to(self.device)
        with torch.no_grad():
            support_feature = self.backbone_net(supp_img_bchw_tensor)
            class_weight_vec = utils.masked_average_pooling(supp_mask_bhw_tensor == 1, support_feature, True)
            class_weight_vec_list.append(class_weight_vec.view((1, -1, 1, 1)))
        classifier_weights = torch.cat(class_weight_vec_list, dim=0) # num_classes x C x 1 x 1
        return classifier_weights
    
    def synthesizer_sample(self, novel_obj_name):
        # Uniformly sample from the memory buffer
        target_id = self.img_name_id_map[novel_obj_name]
        base_img_idx = np.random.choice(self.base_img_candidates)
        assert base_img_idx in self.base_img_candidates
        assert len(self.base_img_candidates) == memory_bank_size
        assert novel_obj_name in self.psuedo_database
        syn_img_chw, syn_mask_hw = self.train_set[base_img_idx]
        if True:
            syn_mask_hw[syn_mask_hw == self.suitcase_idx] = target_id
        # Sample from partial data pool
        # Compute probability for synthesis
        candidate_classes = [c for c in self.psuedo_database.keys() if c != novel_obj_name]
        # Gather some useful numbers
        other_prob = 1
        selected_novel_prob = 1
        num_existing_objects = 0
        num_novel_objects = 2
        assert other_prob >= 0 and other_prob <= 1
        assert selected_novel_prob >= 0 and selected_novel_prob <= 1

        # Synthesize some other objects other than the selected novel object
        if len(self.psuedo_database) > 1 and torch.rand(1) < other_prob:
            # select an old class
            for i in range(num_existing_objects):
                selected_class = np.random.choice(candidate_classes)
                selected_class_id = self.img_name_id_map[selected_class]
                selected_sample = random.choice(self.psuedo_database[selected_class])
                _, _, img_roi, mask_roi = selected_sample
                syn_img_chw, syn_mask_hw = self.copy_and_paste(img_roi, mask_roi, syn_img_chw, syn_mask_hw, selected_class_id)

        # Synthesize selected novel class
        if torch.rand(1) < selected_novel_prob:
            for i in range(num_novel_objects):
                novel_class_id = self.img_name_id_map[novel_obj_name]
                selected_sample = random.choice(self.psuedo_database[novel_obj_name])
                _, _, img_roi, mask_roi = selected_sample
                syn_img_chw, syn_mask_hw = self.copy_and_paste(img_roi, mask_roi, syn_img_chw, syn_mask_hw, novel_class_id)

        return (syn_img_chw, syn_mask_hw)

    def finetune_backbone_one(self, novel_obj_name):
        prv_backbone_net = deepcopy(self.backbone_net).eval()
        self.backbone_net.train()
        self.post_processor.train()

        trainable_params = [
            {"params": self.backbone_net.parameters()},
            {"params": self.post_processor.parameters(), "lr": self.cfg.TASK_SPECIFIC.GIFS.classifier_lr}
        ]

        # Freeze batch norm statistics
        cnt = 0
        for module in self.backbone_net.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, BatchRenorm2d):
                module.weight.requires_grad_(False)
                module.bias.requires_grad_(False)
                module.eval()
                cnt += 1
        
        print("Froze {} BN/BRN layers".format(cnt))

        optimizer = optim.SGD(trainable_params, lr = self.cfg.TASK_SPECIFIC.GIFS.backbone_lr, momentum = 0.9)
        
        max_iter = self.cfg.TASK_SPECIFIC.GIFS.max_iter
        def polynomial_schedule(epoch):
            return (1 - epoch / max_iter)**0.9
        batch_size = self.cfg.TASK_SPECIFIC.GIFS.ft_batch_size

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, polynomial_schedule)

        l2_criterion = nn.MSELoss()

        with trange(1, max_iter + 1, dynamic_ncols=True) as t:
            for iter_i in t:
                image_list = []
                mask_list = []
                for _ in range(batch_size):
                    # synthesis
                    img_chw, mask_hw = self.synthesizer_sample(novel_obj_name)
                    image_list.append(img_chw)
                    mask_list.append(mask_hw)
                data_bchw = torch.stack(image_list).to(self.device).detach()
                target_bhw = torch.stack(mask_list).to(self.device).detach()
                feature = self.backbone_net(data_bchw)
                ori_spatial_res = data_bchw.shape[-2:]
                output = self.post_processor(feature, ori_spatial_res, scale_factor=10)

                # L2 regularization on feature extractor
                with torch.no_grad():
                    # self.vanilla_backbone_net for the base version
                    ori_feature = prv_backbone_net(data_bchw)

                loss = self.criterion(output, target_bhw)

                # Feature extractor regularization + classifier regularization
                regularization_loss = l2_criterion(feature, ori_feature)
                regularization_loss = regularization_loss * self.cfg.TASK_SPECIFIC.GIFS.feature_reg_lambda # hyperparameter lambda
                loss = loss + regularization_loss

                optimizer.zero_grad() # reset gradient
                loss.backward()
                optimizer.step()
                scheduler.step()
                t.set_description_str("Loss: {:.4f}".format(loss.item()))