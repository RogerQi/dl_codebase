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
import vision_hub

from .seg_trainer import seg_trainer
from .fs_incremental_trainer import fs_incremental_trainer
from IPython import embed
import threading

from dataset.open_loris import OpenLORIS_single_sequence_reader

memory_bank_size = 500

class live_continual_seg_trainer(seg_trainer):
    def __init__(self, cfg, backbone_net, post_processor, criterion, dataset_module, device):
        super(live_continual_seg_trainer, self).__init__(cfg, backbone_net, post_processor, criterion, dataset_module, device)

        self.normalizer = tv.transforms.Normalize(mean=self.cfg.DATASET.TRANSFORM.TRAIN.TRANSFORMS_DETAILS.NORMALIZE.mean,
                                    std=self.cfg.DATASET.TRANSFORM.TRAIN.TRANSFORMS_DETAILS.NORMALIZE.sd)
        
        self.psuedo_database = {}
        self.cls_name_id_map = {}
        self.process_pool = {}

        # TODO(roger): for dev purpose we don't use any heuristic now
        np.random.seed(1234)
        self.base_img_candidates = np.random.choice(np.arange(0, len(self.train_set)), replace=False, size=(memory_bank_size,))
        self.class_names = deepcopy(self.train_set.dataset.CLASS_NAMES_LIST)

        self.snapshot_dict = {}

        # Network must either be in training/eval state
        self.model_lock = threading.Lock()
    
    def take_snapshot(self):
        # Save network arch to disk to avoid 
        self.snapshot_dict['backbone_net'] = deepcopy(self.backbone_net.cpu())
        self.snapshot_dict['post_processor'] = deepcopy(self.post_processor.cpu())
        self.snapshot_dict['class_names'] = deepcopy(self.class_names)
        self.snapshot_dict['psuedo_database'] = deepcopy(self.psuedo_database)
        self.snapshot_dict['img_name_id_map'] = deepcopy(self.cls_name_id_map)
        self.backbone_net = self.backbone_net.to(self.device)
        self.post_processor = self.post_processor.to(self.device)
    
    def restore_last_snapshot(self):
        del self.backbone_net
        del self.post_processor
        self.backbone_net = self.snapshot_dict['backbone_net'].to(self.device)
        self.post_processor = self.snapshot_dict['post_processor'].to(self.device)
        self.class_names = self.snapshot_dict['class_names']
        self.psuedo_database = self.snapshot_dict['psuedo_database']
        self.cls_name_id_map = self.snapshot_dict['img_name_id_map']
    
    def test_one(self, device):
        self.backbone_net.eval()
        self.post_processor.eval()
        num_clicks_spent = {}
        my_ritm_segmenter = vision_hub.interactive_seg.ritm_segmenter()
        max_clicks = 20
        iou_thresh = 0.85 # for clicking
        annotation_frame_idx = 20
        offline_inference_iou_dict = {}
        offline_inference_latency_dict = {}
        online_inference_iou_dict = {}
        online_inference_latency_dict = {}
        precision_dict = {}
        for obj_name in ['paper_cutter_01', 'paper_cutter_02', 'paper_cutter_03']:
            num_clicks_spent[obj_name] = 0
            canonical_obj_name = '_'.join(obj_name.split('_')[:-1])
            obj_index = obj_name.split('_')[-1]
            my_seq_reader = OpenLORIS_single_sequence_reader('/data/OpenLORIS', obj_name, 'clutter', 1)
            # Blocking offline evaluation to compute reference metrics
            self.take_snapshot()
            offline_inference_latency_dict[obj_name] = []
            offline_inference_iou_dict[obj_name] = []
            for i in range(len(my_seq_reader)):
                img, mask = my_seq_reader[i]
                img_chw = torch.tensor(img).float().permute((2, 0, 1))
                img_chw = img_chw / 255 # norm to 0-1
                img_chw = self.normalizer(img_chw)
                img_bchw = img_chw.view((1,) + img_chw.shape)
                start_cp = time.time()
                pred_map = self.infer_one(img_bchw).cpu().numpy()[0]
                offline_inference_latency_dict[obj_name].append(time.time() - start_cp)
                label_vis = utils.visualize_segmentation(self.cfg, img_chw, pred_map, self.class_names)
                if canonical_obj_name in self.class_names:
                    binary_seg_metrics = utils.compute_binary_metrics(pred_map == self.class_names.index(canonical_obj_name), mask)
                    iou = binary_seg_metrics['iou']
                else:
                    iou = 0
                offline_inference_iou_dict[obj_name].append(iou)
                cv2.imshow('label', cv2.cvtColor(label_vis, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if i == annotation_frame_idx:
                    # Simulate user inputs. RITM segmeter 
                    provided_mask, num_click = my_ritm_segmenter.auto_eval(img, mask, max_clicks=max_clicks, iou_thresh=iou_thresh)
                    provided_mask = torch.tensor(provided_mask).int()
                    num_clicks_spent[obj_name] += num_click
                    self.novel_adapt_single(img_chw, provided_mask, canonical_obj_name, blocking=True)
            print("[OFFLINE] Avg IoU: {:.4f} pm {:.4f}".format(
                np.mean(offline_inference_iou_dict[obj_name]),
                np.std(offline_inference_iou_dict[obj_name])))
            print("[OFFLINE] Avg latency: {:.4f} w/ std: {:.4f}".format(
                np.mean(offline_inference_latency_dict[obj_name]),
                np.std(offline_inference_latency_dict[obj_name])))
            # Non-blocking online evaluation
            self.restore_last_snapshot()
            online_inference_latency_dict[obj_name] = []
            online_inference_iou_dict[obj_name] = []
            precision_dict[obj_name] = []
            for i in range(len(my_seq_reader)):
                img, mask = my_seq_reader[i]
                img_chw = torch.tensor(img).float().permute((2, 0, 1))
                img_chw = img_chw / 255 # norm to 0-1
                img_chw = self.normalizer(img_chw)
                img_bchw = img_chw.view((1,) + img_chw.shape)
                start_cp = time.time()
                pred_map = self.infer_one(img_bchw).cpu().numpy()[0]
                online_inference_latency_dict[obj_name].append(time.time() - start_cp)
                label_vis = utils.visualize_segmentation(self.cfg, img_chw, pred_map, self.class_names)
                if canonical_obj_name in self.class_names:
                    binary_seg_metrics = utils.compute_binary_metrics(pred_map == self.class_names.index(canonical_obj_name), mask)
                    iou = binary_seg_metrics['iou']
                    precision_dict[obj_name].append(binary_seg_metrics['precision'])
                else:
                    iou = 0
                online_inference_iou_dict[obj_name].append(iou)
                cv2.imshow('label', cv2.cvtColor(label_vis, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if i == annotation_frame_idx:
                    # Simulate user inputs. RITM segmeter 
                    provided_mask, num_click = my_ritm_segmenter.auto_eval(img, mask, max_clicks=max_clicks, iou_thresh=iou_thresh)
                    provided_mask = torch.tensor(provided_mask).int()
                    num_clicks_spent[obj_name] += num_click
                    self.novel_adapt_single(img_chw, provided_mask, canonical_obj_name, blocking=False)
            print("[ONLINE] Avg IoU: {:.4f} pm {:.4f}".format(
                np.mean(online_inference_iou_dict[obj_name]),
                np.std(online_inference_iou_dict[obj_name])))
            print("[ONLINE] Avg precision: {:.4f} pm {:.4f}".format(
                np.mean(precision_dict[obj_name]),
                np.std(precision_dict[obj_name])))
            print("[ONLINE] Avg latency: {:.4f} w/ std: {:.4f}".format(
                np.mean(online_inference_latency_dict[obj_name]), np.std(online_inference_latency_dict[obj_name])))
            print("Main inference completed. Waiting for processes {} to finish".format(self.process_pool.keys()))
            for process_name in self.process_pool:
                self.process_pool[process_name].join()
    
    def live_run(self, device):
        self.infer_on_video()
        print("Main inference completed. Waiting for processes {} to finish".format(self.process_pool.keys()))
        for process_name in self.process_pool:
            self.process_pool[process_name].join()
    
    def infer_on_video(self):
        self.backbone_net.eval()
        self.post_processor.eval()
        video_path = 'corkscrew_01_clutter_1.mp4'
        cap = cv2.VideoCapture(video_path)
        out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640,360))
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
            img_chw = self.normalizer(img_chw)
            img_bchw = img_chw.view((1,) + img_chw.shape)
            start_cp = time.time()
            pred_map = self.infer_one(img_bchw).cpu().numpy()[0]
            print(f"infer time: {time.time() - start_cp}")
            label_vis = utils.visualize_segmentation(self.cfg, img_chw, pred_map, self.class_names)
            cv2.imshow('label', cv2.cvtColor(label_vis, cv2.COLOR_RGB2BGR))
            out.write(cv2.cvtColor(label_vis, cv2.COLOR_RGB2BGR))
            if cnt == 40:
                mask = Image.open('corkscrew_01_clutter_1_frame_40.png')
                mask = np.array(mask)
                mask = torch.tensor(mask).int()
                self.novel_adapt_single(img_chw, mask, 'corkscrew')
            cnt += 1
        out.release()
    
    def infer_one(self, img_bchw):
        self.model_lock.acquire()
        self.backbone_net.eval()
        self.post_processor.eval()
        # Inference
        with torch.no_grad():
            data = img_bchw.to(self.device)
            feature = self.backbone_net(data)
            ori_spatial_res = data.shape[-2:]
            output = self.post_processor(feature, ori_spatial_res)
            pred_map = output.max(dim = 1)[1]
        self.model_lock.release()
        return pred_map
    
    def novel_adapt_single(self, img_chw, mask_hw, obj_name, blocking=True):
        """Adapt to a single image

        Args:
            img (torch.Tensor): Normalized RGB image tensor of shape (3, H, W)
            mask (torch.Tensor): Binary mask of novel object
        """
        self.model_lock.acquire()
        num_existing_class = self.post_processor.pixel_classifier.class_mat.weight.data.shape[0]
        img_roi, mask_roi = utils.crop_partial_img(img_chw, mask_hw)
        if obj_name not in self.psuedo_database:
            self.class_names.append(obj_name)
            self.psuedo_database[obj_name] = [(img_chw, mask_hw, img_roi, mask_roi)]
            new_clf_weights = self.classifier_weight_imprinting_one(img_chw, mask_hw)
            self.post_processor.pixel_classifier.class_mat.weight = torch.nn.Parameter(new_clf_weights)
            self.cls_name_id_map[obj_name] = num_existing_class # 0-indexed shift 1
        else:
            self.psuedo_database[obj_name].append((img_chw, mask_hw, img_roi, mask_roi))
        self.model_lock.release()
        if blocking:
            self.finetune_backbone_one(obj_name)
        else:
            t = threading.Thread(target=self.finetune_backbone_one, args=(obj_name, ))
            t.start()
            self.process_pool['adaptation'] = t

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
        base_img_idx = np.random.choice(self.base_img_candidates)
        assert base_img_idx in self.base_img_candidates
        assert len(self.base_img_candidates) == memory_bank_size
        assert novel_obj_name in self.psuedo_database
        syn_img_chw, syn_mask_hw = self.train_set[base_img_idx]
        # Sample from partial data pool
        # Compute probability for synthesis
        candidate_classes = [c for c in self.psuedo_database.keys() if c != novel_obj_name]
        # Gather some useful numbers
        other_prob = 0.5
        selected_novel_prob = 0.5
        num_existing_objects = 1
        num_novel_objects = 1
        assert other_prob >= 0 and other_prob <= 1
        assert selected_novel_prob >= 0 and selected_novel_prob <= 1

        # Synthesize some other objects other than the selected novel object
        if len(self.psuedo_database) > 1 and torch.rand(1) < other_prob:
            # select an old class
            for i in range(num_existing_objects):
                selected_class = np.random.choice(candidate_classes)
                selected_class_id = self.cls_name_id_map[selected_class]
                selected_sample = random.choice(self.psuedo_database[selected_class])
                _, _, img_roi, mask_roi = selected_sample
                syn_img_chw, syn_mask_hw = utils.copy_and_paste(img_roi, mask_roi, syn_img_chw, syn_mask_hw, selected_class_id)

        # Synthesize selected novel class
        if torch.rand(1) < selected_novel_prob:
            for i in range(num_novel_objects):
                novel_class_id = self.cls_name_id_map[novel_obj_name]
                selected_sample = random.choice(self.psuedo_database[novel_obj_name])
                _, _, img_roi, mask_roi = selected_sample
                syn_img_chw, syn_mask_hw = utils.copy_and_paste(img_roi, mask_roi, syn_img_chw, syn_mask_hw, novel_class_id)

        return (syn_img_chw, syn_mask_hw)

    def finetune_backbone_one(self, novel_obj_name):
        prv_backbone_net = deepcopy(self.backbone_net).eval()
        scaler = torch.cuda.amp.GradScaler()

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
        optimizer.zero_grad() # sanity reset
        
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
                fully_labeled_flag = []
                partial_positive_idx = []
                for _ in range(batch_size):
                    if torch.rand(1) < 0.8:
                        # synthesis
                        img_chw, mask_hw = self.synthesizer_sample(novel_obj_name)
                        image_list.append(img_chw)
                        mask_list.append(mask_hw)
                        fully_labeled_flag.append(True)
                    else:
                        # partially-labeled image
                        img_chw, mask_hw, _, _ = random.choice(self.psuedo_database[novel_obj_name])
                        # TODO: implement proper augmentation
                        img_chw = tr_F.pad(img_chw, [(512 - img_chw.shape[2]) // 2, (512 - img_chw.shape[1]) // 2])
                        mask_hw = tr_F.pad(mask_hw, [(512 - mask_hw.shape[1]) // 2, (512 - mask_hw.shape[0]) // 2])
                        image_list.append(img_chw)
                        mask_list.append(mask_hw)
                        fully_labeled_flag.append(False)
                        partial_positive_idx.append(self.cls_name_id_map[novel_obj_name])
                partial_positive_idx = torch.tensor(partial_positive_idx)
                fully_labeled_flag = torch.tensor(fully_labeled_flag)
                data_bchw = torch.stack(image_list).to(self.device).detach()
                target_bhw = torch.stack(mask_list).to(self.device).detach()
                self.model_lock.acquire()
                self.backbone_net.train()
                self.post_processor.train()
                with torch.cuda.amp.autocast():
                    feature = self.backbone_net(data_bchw)
                    ori_spatial_res = data_bchw.shape[-2:]
                    output = self.post_processor(feature, ori_spatial_res, scale_factor=10)

                    # L2 regularization on feature extractor
                    with torch.no_grad():
                        ori_feature = prv_backbone_net(data_bchw)

                    # Fully-labeled image from memory-replay buffer
                    # TODO: better variable naming!
                    output_logit_bchw = F.softmax(output, dim=1)
                    output_logit_bchw = torch.log(output_logit_bchw)
                    loss = F.nll_loss(output_logit_bchw[fully_labeled_flag], target_bhw[fully_labeled_flag], ignore_index=-1)
                    # loss = self.criterion(output[fully_labeled_flag], target_bhw[fully_labeled_flag])

                    # Partially annotated image propagation using MiB loss
                    if len(partial_positive_idx) > 0:
                        partial_labeled_flag = torch.logical_not(fully_labeled_flag)
                        output_logit_bchw = F.softmax(output[partial_labeled_flag], dim=1)
                        # Reduce to 0/1 using MiB for NLL loss
                        assert output_logit_bchw.shape[0] == len(partial_positive_idx)
                        fg_prob_list = []
                        bg_prob_list = []
                        for b in range(output_logit_bchw.shape[0]):
                            fg_prob_hw = output_logit_bchw[b,partial_positive_idx[b]]
                            bg_prob_chw = output_logit_bchw[b,torch.arange(output_logit_bchw.shape[1]) != partial_positive_idx[b]]
                            bg_prob_hw = torch.sum(bg_prob_chw, dim=0)
                            fg_prob_list.append(fg_prob_hw)
                            bg_prob_list.append(bg_prob_hw)
                        fg_prob_map = torch.stack(fg_prob_list)
                        bg_prob_map = torch.stack(bg_prob_list)
                        reduced_output_bchw = torch.stack([bg_prob_map, fg_prob_map], dim=1)
                        reduced_logit_bchw = torch.log(reduced_output_bchw)

                        # TODO: test sum/mean
                        loss = loss + F.nll_loss(reduced_logit_bchw, target_bhw[partial_labeled_flag], ignore_index=-1)

                    # Feature extractor regularization + classifier regularization
                    regularization_loss = l2_criterion(feature, ori_feature)
                    regularization_loss = regularization_loss * self.cfg.TASK_SPECIFIC.GIFS.feature_reg_lambda # hyperparameter lambda
                    loss = loss + regularization_loss

                optimizer.zero_grad() # reset gradient
                scaler.scale(loss).backward() # loss.backward()
                scaler.step(optimizer) # optimizer.step()
                scaler.update()
                scheduler.step()
                self.model_lock.release()
                t.set_description_str("Loss: {:.4f}".format(loss.item()))
