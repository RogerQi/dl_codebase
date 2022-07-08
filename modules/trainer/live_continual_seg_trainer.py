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
#import vision_hub

from .seg_trainer import seg_trainer
from .fs_incremental_trainer import fs_incremental_trainer
from IPython import embed
import threading

from dataset.scannet_seq_reader import scannet_scene_reader

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

        self.my_aot_segmenter = vision_hub.aotb.aot_segmenter()

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
        offline_inference_latency_dict = {}
        online_inference_latency_dict = {}
        precision_dict = {}
        delay_violation_cnt = 0
        for scene_name in ['scene0050_00', 'scene0565_00', 'scene0462_00', 'scene0144_00', 'scene0593_00']:
            num_clicks_spent[scene_name] = 0
            canonical_obj_name = 'printer'
            my_seq_reader = scannet_scene_reader("/media/eason/My Passport/data/scannet_v2", scene_name)
            found_novel_obj = False
            # Blocking offline evaluation to compute reference metrics
            # self.take_snapshot()
            # offline_inference_latency_dict[scene_name] = []
            # offline_intersection = None
            # offline_union = None
            # first_seen_frame = None
            # for i in range(len(my_seq_reader)):
            #     data_dict = my_seq_reader[i]
            #     img = data_dict['color']
            #     mask = data_dict['semantic_label']
            #     img_chw = torch.tensor(img).float().permute((2, 0, 1))
            #     img_chw = img_chw / 255 # norm to 0-1
            #     img_chw = self.normalizer(img_chw)
            #     img_bchw = img_chw.view((1,) + img_chw.shape)
            #     start_cp = time.time()
            #     pred_map = self.infer_one(img_bchw).cpu().numpy()[0]
            #     offline_inference_latency_dict[scene_name].append(time.time() - start_cp)
            #     label_vis = utils.visualize_segmentation(self.cfg, img_chw, pred_map, self.class_names)
            #     if canonical_obj_name in self.class_names:
            #         # Seen this object before, eval
            #         intersection, union = utils.compute_iu(pred_map, mask, num_classes=22, fg_only=True)
            #         if offline_intersection is None:
            #             offline_intersection = intersection
            #             offline_union = union
            #         else:
            #             offline_intersection += intersection
            #             offline_union += union
            #     # cv2.imshow('label', cv2.cvtColor(label_vis, cv2.COLOR_RGB2BGR))
            #     # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     #     break
            #     if 21 in mask and first_seen_frame is None:
            #         first_seen_frame = i
            #     if first_seen_frame is not None and i == first_seen_frame + 30 and scene_name != 'scene0593_00':
            #         assert 21 in mask
            #         assert np.sum(mask == 21) > 2000 # only objects large enough will be considered
            #         # select relevant instance
            #         inst_idx_list, inst_pixel_cnt = np.unique(data_dict['inst_label'][data_dict['semantic_label'] == 21], return_counts=True)
            #         selected_instance = inst_idx_list[np.argmax(inst_pixel_cnt)]
            #         instance_mask = (data_dict['inst_label'] == selected_instance).astype(np.uint8)
            #         # Simulate user inputs. RITM segmeter 
            #         provided_mask, num_click = my_ritm_segmenter.auto_eval(img, instance_mask, max_clicks=max_clicks, iou_thresh=iou_thresh)
            #         provided_mask = torch.tensor(provided_mask).int()
            #         self.novel_adapt_single(img_chw, provided_mask, canonical_obj_name, blocking=True)
            # print("[OFFLINE] Avg IoU: {:.4f}".format(np.mean(offline_intersection[-1] / (offline_union[-1] + 1e-10))))
            # print("[OFFLINE] Avg latency: {:.4f} w/ std: {:.4f}".format(
            #     np.mean(offline_inference_latency_dict[scene_name]),
            #     np.std(offline_inference_latency_dict[scene_name])))
            # # Non-blocking online evaluation
            # self.restore_last_snapshot()
            online_inference_latency_dict[scene_name] = []
            online_intersection = None
            online_union = None
            first_seen_frame = None
            for i in range(len(my_seq_reader)):
                data_dict = my_seq_reader[i]
                img = data_dict['color']
                mask = data_dict['semantic_label']
                img_chw = torch.tensor(img).float().permute((2, 0, 1))
                img_chw = img_chw / 255 # norm to 0-1
                img_chw = self.normalizer(img_chw)
                img_bchw = img_chw.view((1,) + img_chw.shape)
                start_cp = time.time()
                pred_map = self.infer_one(img_bchw).cpu().numpy()[0]
                inference_latency = time.time() - start_cp
                online_inference_latency_dict[scene_name].append(inference_latency)

                # Overwrite pred_map with VOS
                if found_novel_obj: # if found novel obj, start short-term online eval with VOS method.
                    # read_fn = f'/home/roger/reproduction/aot-benchmark/demo_output/pred_masks/{scene_name}/frame-{str(i).zfill(6)}.png'
                    # if os.path.exists(read_fn):
                    #     vos_pred = np.array(Image.open(read_fn)).astype(np.uint8)
                    #     vos_pred = cv2.resize(vos_pred, (640, 480), interpolation=cv2.INTER_NEAREST)
                    #     pred_map[vos_pred == 1] = 21
                    pred_map = self.infer_w_aot(pred_map, img)

                label_vis = utils.visualize_segmentation(self.cfg, img_chw, pred_map, self.class_names)
                if canonical_obj_name in self.class_names:
                    # Seen this object before, eval
                    intersection, union = utils.compute_iu(pred_map, mask, num_classes=22, fg_only=True)
                    if online_intersection is None:
                        online_intersection = intersection
                        online_union = union
                    else:
                        online_intersection += intersection
                        online_union += union
                cv2.imshow('label', cv2.cvtColor(label_vis, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if 21 in mask and first_seen_frame is None:
                    first_seen_frame = i
                if first_seen_frame is not None and i == first_seen_frame + 30 and scene_name != 'scene0593_00':
                    assert 21 in mask
                    assert np.sum(mask == 21) > 2000 # only objects large enough will be considered
                    # select relevant instance
                    inst_idx_list, inst_pixel_cnt = np.unique(data_dict['inst_label'][data_dict['semantic_label'] == 21], return_counts=True)
                    selected_instance = inst_idx_list[np.argmax(inst_pixel_cnt)]
                    instance_mask = (data_dict['inst_label'] == selected_instance).astype(np.uint8)
                    # Simulate user inputs. RITM segmeter 
                    provided_mask, num_click = my_ritm_segmenter.auto_eval(img, instance_mask, max_clicks=max_clicks, iou_thresh=iou_thresh)
                    num_clicks_spent[scene_name] += num_click
                    provided_mask = torch.tensor(provided_mask).long()

                    # set aot segmenter reference frame.
                    found_novel_obj = True
                    self.my_aot_segmenter.reset_engine()
                    label = deepcopy(provided_mask)
                    # label[label == 1] = 21
                    self.my_aot_segmenter.add_reference_frame(img, label.cpu().numpy())
                    
                    self.novel_adapt_single(img_chw, provided_mask, canonical_obj_name, blocking=False)
            print("[ONLINE] Avg IoU: {:.4f}".format(np.mean(online_intersection[-1] / (online_union[-1] + 1e-10))))
            print("[ONLINE] Avg latency: {:.4f} w/ std: {:.4f}".format(
                np.mean(online_inference_latency_dict[scene_name]), np.std(online_inference_latency_dict[scene_name])))
            print("Main inference completed. Waiting for processes {} to finish".format(self.process_pool.keys()))
            for process_name in self.process_pool:
                self.process_pool[process_name].join()
            # Add VOS tracked examples to data pool
            # if scene_name != 'scene0593_00':
            #     valid_mask_dir = os.path.join('/home/roger/dl_codebase/june17_temp', scene_name)
            #     all_masks = os.listdir(valid_mask_dir)
            #     for mask_fn in all_masks:
            #         mask_idx = int(mask_fn.split('-')[1].split('.')[0])
            #         mask_path = os.path.join(valid_mask_dir, mask_fn)
            #         vos_mask = torch.tensor(np.array(Image.open(mask_path)).astype(np.uint8))
            #         data_dict = my_seq_reader[mask_idx]
            #         img = data_dict['color']
            #         img_chw = torch.tensor(img).float().permute((2, 0, 1))
            #         img_chw = img_chw / 255 # norm to 0-1
            #         img_chw = self.normalizer(img_chw)
            #         img_roi, mask_roi = utils.crop_partial_img(img_chw, vos_mask)
            #         # FIXME: figure out why this happens?
            #         if img_roi.shape[1] == 0 or img_roi.shape[2] == 0:
            #             continue
            #         self.psuedo_database['printer'].append((img_chw, vos_mask, img_roi, mask_roi))
            #     print("Adapting to VOS samples")
            #     self.finetune_backbone_one('printer')
        print("Total clicks expanded: {}".format(num_clicks_spent))
        print(f"Total delay violation: {delay_violation_cnt}")
        print("eval on old dataset to test catastrophic forgetting")
        class_iou, pixel_acc = self.eval_on_loader(self.val_loader, 22)
        print("Base IoU after adaptation")
        print(np.mean(class_iou[:-1]))
    
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
    

    def infer_w_aot(self, pred_map, img):
        if self.my_aot_segmenter.frame_cnt == 0:
            print("havn't add reference frame for aot segmenter")
            return
        pred_label = self.my_aot_segmenter.propagate_one_frame(img)
        print(pred_label.shape)
        pred_map[pred_label == 1] = 21
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
                    # FIXME: check if there is bug
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
        
    def prepare_for_eval(self):
        self.backbone_net.eval()
        self.post_processor.eval()
        
