import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from copy import deepcopy
import cv2
from numpy.core.defchararray import replace
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

from .sequential_GIFS_seg_trainer import sequential_GIFS_seg_trainer
from dataset.special_loader import get_fs_seg_loader

from IPython import embed

def harmonic_mean(base_iou, novel_iou):
    return 2 / (1. / base_iou + 1. / novel_iou)

def copy_and_paste(novel_img_chw, novel_mask_hw, base_img_chw, base_mask_hw, mask_id):
    # Horizontal Flipping
    if torch.rand(1) < 0.5:
        novel_img_chw = tr_F.hflip(novel_img_chw)
        novel_mask_hw = tr_F.hflip(novel_mask_hw)

    # Random Translation
    h, w = novel_mask_hw.shape
    # Biased sampling to select novel class more often
    if base_mask_hw.shape[0] > h and base_mask_hw.shape[1] > w:
        paste_x = torch.randint(low=0, high=base_mask_hw.shape[1] - w, size=(1,))
        paste_y = torch.randint(low=0, high=base_mask_hw.shape[0] - h, size=(1,))
    else:
        paste_x = 0
        paste_y = 0
    
    base_img_chw[:,paste_y:paste_y+h,paste_x:paste_x+w][:,novel_mask_hw] = novel_img_chw[:,novel_mask_hw]
    base_mask_hw[paste_y:paste_y+h,paste_x:paste_x+w][novel_mask_hw] = mask_id

    img_chw = base_img_chw
    mask_hw = base_mask_hw

    return (img_chw, mask_hw)

class fs_incremental_trainer(sequential_GIFS_seg_trainer):
    def __init__(self, cfg, backbone_net, post_processor, criterion, dataset_module, device):
        super(fs_incremental_trainer, self).__init__(cfg, backbone_net, post_processor, criterion, dataset_module, device)
        
        self.partial_data_pool = {}
        self.demo_pool = {}

        self.base_img_candidates = np.arange(0, len(self.train_set))
        self.base_img_candidates = np.random.choice(self.base_img_candidates, replace=False, size=(500,))
    
    def live_run(self, device):
        self.backbone_net.eval()
        self.post_processor.eval()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        
        image_to_tensor = torchvision.transforms.ToTensor()
        normalizer = torchvision.transforms.Normalize(self.cfg.DATASET.TRANSFORM.TEST.TRANSFORMS_DETAILS.NORMALIZE.mean,
                                    self.cfg.DATASET.TRANSFORM.TEST.TRANSFORMS_DETAILS.NORMALIZE.sd)
        
        base_class_name_list = [i for i in self.train_set.dataset.CLASS_NAMES_LIST if i not in self.train_set.dataset.NOVEL_CLASSES_LIST]
        
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            assert frame.shape == (480, 640, 3)
            H, W, C = frame.shape
            assert W >= H
            strip_size = (W - H) // 2
            frame = frame[:, strip_size:-strip_size, :]
            H, W, C = frame.shape
            assert H == W
            frame = cv2.resize(frame, (480, 480), interpolation = cv2.INTER_LINEAR)
            # Model Inference
            data = image_to_tensor(frame) # 3 x H x W
            # Image in OpenCV are stored as BGR. Need to convert to RGB
            with torch.no_grad():
                assert data.shape[0] == 3
                tmp = data[0].clone()
                data[0] = data[2]
                data[2] = tmp
            assert data.shape == (C, H, W)
            assert data.min() >= 0
            assert data.max() <= 1
            data = normalizer(data)
            data = data.view((1,) + data.shape) # B x 3 x H x W
            with torch.no_grad():
                # Forward Pass
                data = data.to(device)
                feature = self.backbone_net(data)
                ori_spatial_res = data.shape[-2:]
                output = self.post_processor(feature, ori_spatial_res)
                # Visualization
                pred_map = output.max(dim = 1)[1]
                assert pred_map.shape[0] == 1
                pred_np = pred_map[0].cpu().numpy()
                predicted_label = utils.visualize_segmentation(self.cfg, data[0], pred_np, base_class_name_list)
            # Display the resulting frame
            cv2.imshow('raw_image', frame)
            cv2.imshow('predicted_label', predicted_label)
            key_press = cv2.waitKey(1)
            if key_press == ord('q'):
                # Quit!
                break
            elif key_press == ord('s'):
                # Save image for segmentation!
                cv2.imwrite("/tmp/temp.jpg", frame)
                obj_name = input("Name of the novel object: ")
                os.system('cd /home/roger/reproduction/fcanet && python3 annotator.py --backbone resnet --input /tmp/temp.jpg --output /tmp/temp_mask.png --sis')
                provided_mask = cv2.imread('/tmp/temp_mask.png', cv2.IMREAD_UNCHANGED)
                if obj_name not in base_class_name_list:
                    base_class_idx = [i for i in range(len(base_class_name_list))]
                    novel_obj_id = max(base_class_idx) + 1
                    novel_class_idx = [novel_obj_id]
                    provided_mask = (provided_mask == 255).astype(np.uint8) * novel_obj_id
                    provided_mask = torch.tensor(provided_mask).view((1,) + provided_mask.shape).cuda() # 1 x H x W
                    # MAP on feature
                    self.prv_backbone_net = deepcopy(self.backbone_net)
                    self.prv_post_processor = deepcopy(self.post_processor)
                    
                    self.prv_backbone_net.eval()
                    self.prv_post_processor.eval()
                    supp_img_bchw = data.cpu()
                    supp_mask_bhw = provided_mask.cpu()

                    assert novel_obj_id not in self.demo_pool
                    self.demo_pool[novel_obj_id] = [(supp_img_bchw, supp_mask_bhw)]

                    # Novel adaption
                    max_cls = max(max(base_class_idx), max(novel_class_idx)) + 1
                    self.post_processor = classifier.dispatcher(self.cfg, self.feature_shape, num_classes=max_cls)
                    self.post_processor = self.post_processor.to(self.device)
                    # Aggregate weights
                    aggregated_weights = self.classifier_weight_imprinting(base_class_idx, novel_class_idx, supp_img_bchw, supp_mask_bhw)
                    self.post_processor.pixel_classifier.class_mat.weight.data = aggregated_weights

                    self.finetune_backbone(base_class_idx, novel_class_idx, supp_img_bchw, supp_mask_bhw)
                
                    base_class_name_list.append(obj_name)
                else:
                    novel_obj_id = base_class_name_list.index(obj_name)
                    novel_class_idx = [novel_obj_id]
                    provided_mask = (provided_mask == 255).astype(np.uint8) * novel_obj_id
                    provided_mask = torch.tensor(provided_mask).view((1,) + provided_mask.shape).cuda() # 1 x H x W
                    # MAP on feature
                    self.prv_backbone_net = deepcopy(self.backbone_net)
                    # Delete existing one
                    clf_subset = torch.ones(self.post_processor.pixel_classifier.class_mat.weight.data.shape[0]).bool()
                    clf_subset[novel_class_idx] = False
                    self.post_processor.pixel_classifier.class_mat.weight.data = self.post_processor.pixel_classifier.class_mat.weight.data[clf_subset]
                    self.prv_post_processor = deepcopy(self.post_processor)
                    
                    self.prv_backbone_net.eval()
                    self.prv_post_processor.eval()

                    # Aggregate
                    supp_img_bchw = data.cpu()
                    supp_mask_bhw = provided_mask.cpu()

                    # TODO: use stacking for faster OP
                    for img, mask in self.demo_pool[novel_obj_id]:
                        supp_img_bchw = torch.cat([supp_img_bchw, img])
                        supp_mask_bhw = torch.cat([supp_mask_bhw, mask])

                    self.partial_data_pool = {}
                    # Novel adaption
                    self.novel_adapt(base_class_idx, novel_class_idx, supp_img_bchw, supp_mask_bhw)
        
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
    
    def synthesizer_sample(self, novel_obj_id):
        # Sample from complete data pool (base dataset)
        if True:
            base_img_idx = np.random.choice(self.base_img_candidates)
        else:
            base_img_idx = np.random.randint(0, len(self.train_set))
        syn_img_chw, syn_mask_hw = self.train_set[base_img_idx]

        # Sample from partial data pool
        if len(self.partial_data_pool) > 1 and torch.rand(1) < 0.42:
            # select an old class
            candidate_classes = [c for c in self.partial_data_pool.keys() if c != novel_obj_id]
            num_objs = np.random.choice([1, 2, 3], p=(0, 1, 0))
            for i in range(num_objs):
                selected_class = np.random.choice(candidate_classes)
                selected_sample = random.choice(self.partial_data_pool[selected_class])
                img_chw, mask_hw = selected_sample
                syn_img_chw, syn_mask_hw = copy_and_paste(img_chw, mask_hw, syn_img_chw, syn_mask_hw, selected_class)

        if torch.rand(1) < 0.8: 
            num_objs = np.random.choice([1, 2, 3], p=(0, 1, 0))
            for i in range(num_objs):
                selected_sample = random.choice(self.partial_data_pool[novel_obj_id])
                img_chw, mask_hw = selected_sample
                syn_img_chw, syn_mask_hw = copy_and_paste(img_chw, mask_hw, syn_img_chw, syn_mask_hw, novel_obj_id)

        return (syn_img_chw, syn_mask_hw)

    def finetune_backbone(self, base_class_idx, novel_class_idx, supp_img_bchw, supp_mask_bhw):
        assert self.prv_backbone_net is not None
        assert self.prv_post_processor is not None
        assert len(novel_class_idx) == 1

        novel_obj_id = novel_class_idx[0]

        for b in range(supp_img_bchw.shape[0]):
            novel_img_chw = supp_img_bchw[b]
            mask_hw = supp_mask_bhw[b]
            novel_mask_hw = (mask_hw == novel_obj_id)

            novel_mask_hw_np = novel_mask_hw.numpy().astype(np.uint8)

            # RETR_EXTERNAL to keep online the outer contour
            contours, _ = cv2.findContours(novel_mask_hw_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for ctr in contours:
                x, y, w, h = cv2.boundingRect(ctr)
                if novel_obj_id not in self.partial_data_pool:
                    self.partial_data_pool[novel_obj_id] = []
                mask_roi = novel_mask_hw[y:y+h,x:x+w]
                img_roi = novel_img_chw[:,y:y+h,x:x+w]
                self.partial_data_pool[novel_obj_id].append((img_roi, mask_roi))

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
                image_list = []
                mask_list = []
                for _ in range(batch_size):
                    img_chw, mask_hw = self.synthesizer_sample(novel_obj_id)
                    image_list.append(img_chw)
                    mask_list.append(mask_hw)
                data_bchw = torch.stack(image_list).cuda()
                target_bhw = torch.stack(mask_list).cuda()
                feature = self.backbone_net(data_bchw)
                ori_spatial_res = data_bchw.shape[-2:]
                output = self.post_processor(feature, ori_spatial_res, scale_factor=10)

                # L2 regularization on feature extractor
                with torch.no_grad():
                    ori_feature = self.vanilla_backbone_net(data_bchw)
                    ori_logit = self.vanilla_post_processor(ori_feature, ori_spatial_res, scale_factor=10)

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
                clf_loss = l2_criterion(output[:,self.vanilla_base_class_idx,:,:], ori_logit) * self.cfg.TASK_SPECIFIC.GIFS.classifier_reg_lambda
                loss = loss + clf_loss

                optimizer.zero_grad() # reset gradient
                loss.backward()
                optimizer.step()
                scheduler.step()
                t.set_description_str("Loss: {:.4f}".format(loss.item()))
