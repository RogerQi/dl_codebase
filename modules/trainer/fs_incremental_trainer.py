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

from .sequential_GIFS_seg_trainer import sequential_GIFS_seg_trainer
from IPython import embed

def harmonic_mean(base_iou, novel_iou):
    return 2 / (1. / base_iou + 1. / novel_iou)

class kd_criterion(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x_pred, x_ref):
        # x_pred and x_ref are unnormalized
        x_pred = F.softmax(x_pred, dim=1)
        x_ref = F.softmax(x_ref, dim=1)
        element_wise_loss = x_ref * torch.log(x_pred)
        return -torch.mean(element_wise_loss)

memory_bank_size = 500

class fs_incremental_trainer(sequential_GIFS_seg_trainer):
    def __init__(self, cfg, backbone_net, post_processor, criterion, dataset_module, device):
        super(fs_incremental_trainer, self).__init__(cfg, backbone_net, post_processor, criterion, dataset_module, device)
        
        self.partial_data_pool = {}
        self.blank_bank = {}
        self.demo_pool = {}

        self.train_set_vanilla_label = dataset_module.get_train_set_vanilla_label(cfg)
        self.vanilla_train_set = dataset_module.get_vanilla_train_set_vanilla_label(cfg)
        self.train_set_unaug = dataset_module.get_unaug_train_set(cfg)

        self.context_aware_prob = self.cfg.TASK_SPECIFIC.GIFS.context_aware_sampling_prob

        assert self.context_aware_prob >= 0
        assert self.context_aware_prob <= 1
    
    def construct_baseset(self):
        baseset_type = self.cfg.TASK_SPECIFIC.GIFS.baseset_type
        baseset_folder = f"save_{self.cfg.name}"
        if baseset_type == 'random':
            print(f"construct {baseset_type} baseset for {self.cfg.name}")
            examplar_list = np.arange(0, len(self.train_set))
        elif baseset_type in ['far', 'close', 'far_close', 'uniform_interval']:
            print(f"construct {baseset_type} baseset for {self.cfg.name}")
            base_id_list = self.train_set.dataset.get_label_range()
            if not os.path.exists(f"{baseset_folder}/similarity_dic"):
                self.prv_backbone_net = deepcopy(self.backbone_net)
                self.prv_post_processor = deepcopy(self.post_processor)
                
                self.prv_backbone_net.eval()
                self.prv_post_processor.eval()

                print(f"self.base_id_list contains {base_id_list}")
                base_id_set = set(base_id_list)
                
                mean_weight_dic = {}
                similarity_dic = {}

                # Get the mean weight of each class
                for c in base_id_list:
                    mean_weight_dic[c] = self.prv_post_processor.pixel_classifier.class_mat.weight.data[c]
                    mean_weight_dic[c] = mean_weight_dic[c].reshape((-1))
                    mean_weight_dic[c] = mean_weight_dic[c].cpu().unsqueeze(0)
                    similarity_dic[c] = []
 
                # Maintain a m-size heap to store the top m images of each class
                for i in tqdm(range(len(self.train_set_unaug))):
                    img, mask = self.train_set_unaug[i]
                    class_list = torch.unique(mask).tolist()
                    img_tensor = torch.stack([img]).to(self.device)
                    mask_tensor = torch.stack([mask]).to(self.device)
                    for c in class_list:
                        if c not in base_id_set:
                            continue
                        with torch.no_grad():
                            img_feature = self.prv_backbone_net(img_tensor)
                            img_weight = utils.masked_average_pooling(mask_tensor == c, img_feature, True)
                        img_weight = img_weight.cpu().unsqueeze(0)
                        similarity = F.cosine_similarity(img_weight, mean_weight_dic[c])
                        similarity_dic[c].append((similarity, i))
                for c in base_id_list:
                    similarity_dic[c] = sorted(similarity_dic[c], key=lambda x:x[0])
                if not os.path.exists(baseset_folder):   
                    os.makedirs(baseset_folder)  
                torch.save(similarity_dic, f"{baseset_folder}/similarity_dic")
            else:
                print("load similarity_dic")
                similarity_dic = torch.load(f"{baseset_folder}/similarity_dic")
            m = (memory_bank_size // len(base_id_list)) * 2
            if baseset_type == 'far':
                m_far = m
                m_close = 0
            elif baseset_type == 'close':
                m_far = 0
                m_close = m
            elif baseset_type == 'far_close':
                m_close = m // 2
                m_far = m // 2
            elif baseset_type == 'uniform_interval':
                m_interval = m
            else:
                raise NotImplementedError
            # Combine all the top images of each class by set union
            examplar_set = set()
            for c in base_id_list:
                if baseset_type == 'close' or baseset_type == 'far_close':
                    assert m_close <= len(similarity_dic[c])
                    close_list = similarity_dic[c][-m_close:]
                    class_examplar_list = [i for _, i in close_list]
                    examplar_set = examplar_set.union(class_examplar_list)
                if baseset_type == 'far' or baseset_type == 'far_close':
                    assert m_far <= len(similarity_dic[c])
                    far_list = similarity_dic[c][:m_far]
                    class_examplar_list = [i for _, i in far_list]
                    examplar_set = examplar_set.union(class_examplar_list)
                if baseset_type == 'uniform_interval':
                    assert m_interval <= len(similarity_dic[c]), f"Need {m_interval} from {len(similarity_dic[c])}"
                    interval_size = len(similarity_dic[c]) // m_interval
                    class_examplar_list = []
                    for i in range(m_interval):
                        int_start = interval_size * i
                        assert int_start < len(similarity_dic[c])
                        int_end = min(interval_size * (i + 1), len(similarity_dic[c]))
                        sample_tuple = similarity_dic[c][np.random.randint(low=int_start, high=int_end)]
                        class_examplar_list.append(sample_tuple[1]) # first element is similarity
                    examplar_set = examplar_set.union(class_examplar_list)
            examplar_list = sorted(list(examplar_set))
        else:
            raise AssertionError('invalid baseset_type', baseset_type)
        print(f"total number of examplar_list {len(examplar_list)}")
        return np.random.choice(examplar_list, replace=False, size=(memory_bank_size,))
    
    def test_one(self, device, num_runs=5):
        if self.cfg.TASK_SPECIFIC.GIFS.num_runs != -1:
            num_runs = self.cfg.TASK_SPECIFIC.GIFS.num_runs
        self.base_img_candidates = self.construct_baseset()
        if self.context_aware_prob > 0:
            self.scene_model_setup()
        sequential_GIFS_seg_trainer.test_one(self, device, num_runs)
    
    def synthesizer_sample(self, novel_obj_id):
        # Sample an image from base memory bank
        if torch.rand(1) < self.context_aware_prob:
            # Context-aware sampling from a contextually-similar subset of the memory replay buffer
            memory_buffer_idx = np.random.choice(self.context_similar_map[novel_obj_id])
            base_img_idx = self.base_img_candidates[memory_buffer_idx]
        else:
            # Uniformly sample from the memory buffer
            base_img_idx = np.random.choice(self.base_img_candidates)
        assert base_img_idx in self.base_img_candidates
        assert len(self.base_img_candidates) == memory_bank_size
        syn_img_chw, syn_mask_hw = self.train_set_vanilla_label[base_img_idx]
        # Sample from partial data pool
        # Compute probability for synthesis
        candidate_classes = [c for c in self.partial_data_pool.keys() if c != novel_obj_id]
        # Gather some useful numbers
        num_base_classes = len(self.train_set_vanilla_label.dataset.get_label_range())
        num_novel_classes = len(self.partial_data_pool)
        total_classes = num_base_classes + num_novel_classes
        num_novel_instances = len(self.partial_data_pool[novel_obj_id])
        for k in candidate_classes:
            assert len(self.partial_data_pool[k]) == num_novel_instances, "every class is expected to have $numShot$ samples"
        if self.cfg.TASK_SPECIFIC.GIFS.probabilistic_synthesis_strat == 'vRFS':
            t = 1. / total_classes
            f_n = num_novel_instances / memory_bank_size
            r_e = max(1, np.sqrt(t / f_n))
            r_n = r_e
            other_prob = (r_e + r_n) / (r_e + r_n + 1 + r_n)
            selected_novel_prob = (r_n + r_n) / (r_e + r_n + 1 + r_n)
            num_existing_objects = 2
            num_novel_objects = 2
        elif self.cfg.TASK_SPECIFIC.GIFS.probabilistic_synthesis_strat == 'always':
            other_prob = 0
            selected_novel_prob = 1
            num_existing_objects = 0
            num_novel_objects = 2
        elif self.cfg.TASK_SPECIFIC.GIFS.probabilistic_synthesis_strat == 'CAS':
            other_prob = 1. / total_classes
            selected_novel_prob = 1. / total_classes
            num_existing_objects = 1
            num_novel_objects = 1
        else:
            raise NotImplementedError("Unknown probabilistic synthesis strategy: {}".format(self.cfg.TASK_SPECIFIC.GIFS.probabilistic_synthesis_strat))
        assert other_prob >= 0 and other_prob <= 1
        assert selected_novel_prob >= 0 and selected_novel_prob <= 1

        # Synthesize some other objects other than the selected novel object
        if len(self.partial_data_pool) > 1 and torch.rand(1) < other_prob:
            # select an old class
            for i in range(num_existing_objects):
                selected_class = np.random.choice(candidate_classes)
                selected_sample = random.choice(self.partial_data_pool[selected_class])
                img_chw, mask_hw = selected_sample
                syn_img_chw, syn_mask_hw = self.copy_and_paste(img_chw, mask_hw, syn_img_chw, syn_mask_hw, selected_class)

        # Synthesize selected novel class
        if torch.rand(1) < selected_novel_prob:
            for i in range(num_novel_objects):
                selected_sample = random.choice(self.partial_data_pool[novel_obj_id])
                img_chw, mask_hw = selected_sample
                syn_img_chw, syn_mask_hw = self.copy_and_paste(img_chw, mask_hw, syn_img_chw, syn_mask_hw, novel_obj_id)

        return (syn_img_chw, syn_mask_hw)
    
    def get_scene_embedding(self, img):
        '''
        img: normalized image tensor of shape CHW
        '''
        assert len(img.shape) == 3 # CHW
        img = img.view((1,) + img.shape)
        if img.shape[2] != 224:
            # Scene model is trained using 224 x 224 size
            img = F.interpolate(img, size = (224, 224), mode = 'bilinear')
        with torch.no_grad():
            scene_embedding = self.scene_model(img)
            scene_embedding = torch.flatten(scene_embedding, start_dim = 1)
            # normalize to unit vector
            norm = torch.norm(scene_embedding, p=2, dim =1).unsqueeze(1).expand_as(scene_embedding) # norm
            scene_embedding = scene_embedding.div(norm+ 1e-5)
        return scene_embedding.squeeze()
    
    def scene_model_setup(self):
        # Load torchscript
        self.scene_model = torch.jit.load('/data/cvpr2022/vgg16_scene_net.pt')
        self.scene_model = self.scene_model.to(self.device)
        # Compute feature vectors for data in the pool
        self.base_pool_cos_embeddings = []
        for base_data_idx in self.base_img_candidates:
            img_chw, _ = self.vanilla_train_set[base_data_idx]
            img_chw = img_chw.to(self.device)
            scene_embedding = self.get_scene_embedding(img_chw)
            assert len(scene_embedding.shape) == 1
            self.base_pool_cos_embeddings.append(scene_embedding)
        self.base_pool_cos_embeddings = torch.stack(self.base_pool_cos_embeddings)
    
    def copy_and_paste(self, novel_img_chw, novel_mask_hw, base_img_chw, base_mask_hw, mask_id):
        base_img_chw = base_img_chw.clone()
        base_mask_hw = base_mask_hw.clone()
        # Horizontal Flipping
        if torch.rand(1) < 0.5:
            novel_img_chw = tr_F.hflip(novel_img_chw)
            novel_mask_hw = tr_F.hflip(novel_mask_hw)
        
        # Parameters for random resizing
        scale = np.random.uniform(0.1, 2.0)
        src_h, src_w = novel_mask_hw.shape
        if src_h * scale > base_mask_hw.shape[0]:
            scale = base_mask_hw.shape[0] / src_h
        if src_w * scale > base_mask_hw.shape[1]:
            scale = base_mask_hw.shape[1] / src_w
        target_H = int(src_h * scale)
        target_W = int(src_w * scale)
        if target_H == 0: target_H = 1
        if target_W == 0: target_W = 1
        # apply
        novel_img_chw = tr_F.resize(novel_img_chw, (target_H, target_W))
        novel_mask_hw = novel_mask_hw.view((1,) + novel_mask_hw.shape)
        novel_mask_hw = tr_F.resize(novel_mask_hw, (target_H, target_W), interpolation=tv.transforms.InterpolationMode.NEAREST)
        novel_mask_hw = novel_mask_hw.view(novel_mask_hw.shape[1:])

        # Random Translation
        h, w = novel_mask_hw.shape
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

    def finetune_backbone(self, base_class_idx, novel_class_idx, support_set):
        assert self.prv_backbone_net is not None
        assert self.prv_post_processor is not None

        print(f"Adapting to novel id {novel_class_idx}")

        if self.context_aware_prob > 0:
            for novel_obj_id in novel_class_idx:
                assert novel_obj_id in support_set
                for idx in support_set[novel_obj_id]:
                    novel_img_chw, mask_hw = self.continual_vanilla_train_set[idx]
                    
                    # Compute cosine embedding
                    if self.context_aware_prob > 0:
                        scene_embedding = self.get_scene_embedding(novel_img_chw.to(self.device))
                        scene_embedding = scene_embedding.view((1,) + scene_embedding.shape)
                        similarity_score = F.cosine_similarity(scene_embedding, self.base_pool_cos_embeddings)
                        base_candidates = torch.argsort(similarity_score)[-int(0.1 * self.base_pool_cos_embeddings.shape[0]):] # Indices array
                        if novel_obj_id not in self.context_similar_map:
                            self.context_similar_map[novel_obj_id] = list(base_candidates.cpu().numpy())
                        else:
                            self.context_similar_map[novel_obj_id] += list(base_candidates.cpu().numpy())
            for c in self.context_similar_map:
                self.context_similar_map[c] = list(set(self.context_similar_map[c]))

        for novel_obj_id in novel_class_idx:
            assert novel_obj_id in support_set
            for idx in support_set[novel_obj_id]:
                novel_img_chw, mask_hw = self.continual_vanilla_train_set[idx]
                img_roi, mask_roi = utils.crop_partial_img(novel_img_chw, mask_hw, cls_id=novel_obj_id)
                assert mask_roi.shape[0] > 0 and mask_roi.shape[1] > 0
                self.partial_data_pool[novel_obj_id].append((img_roi, mask_roi))

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
        my_kd_criterion = kd_criterion()

        # Save vanilla self.post_processor (after MAP) for prototype distillation
        post_processor_distillation_ref = deepcopy(self.post_processor)

        with trange(1, max_iter + 1, dynamic_ncols=True) as t:
            for iter_i in t:
                image_list = []
                mask_list = []
                for _ in range(batch_size):
                    if True:
                        # synthesis
                        novel_obj_id = random.choice(novel_class_idx)
                        img_chw, mask_hw = self.synthesizer_sample(novel_obj_id)
                        image_list.append(img_chw)
                        mask_list.append(mask_hw)
                    else:
                        # full mask
                        chosen_cls = random.choice(list(novel_class_idx))
                        idx = random.choice(support_set[chosen_cls])
                        img_chw, mask_hw = self.continual_aug_train_set[idx]
                        if False:
                            # Mask non-novel portion using pseudo labels
                            with torch.no_grad():
                                data_bchw = img_chw.to(self.device)
                                data_bchw = data_bchw.view((1,) + data_bchw.shape)
                                feature = self.prv_backbone_net(data_bchw)
                                ori_spatial_res = data_bchw.shape[-2:]
                                output = self.prv_post_processor(feature, ori_spatial_res, scale_factor=10)
                            tmp_target_hw = output.max(dim = 1)[1].cpu()[0]
                            for novel_obj_id in support_set.keys():
                                novel_mask = torch.zeros_like(mask_hw)
                                novel_mask = torch.logical_or(novel_mask, mask_hw == novel_obj_id)
                            tmp_target_hw[novel_mask] = mask_hw[novel_mask]
                            mask_hw = tmp_target_hw
                        if False:
                            # Copy-paste augmentation from other support images
                            copy_cls = random.choice(list(novel_class_idx))
                            copy_idx = random.choice(support_set[copy_cls])
                            copy_img_chw, copy_mask_hw = self.continual_vanilla_train_set[copy_idx]
                            img_chw, mask_hw = self.copy_and_paste(copy_img_chw, copy_mask_hw == copy_cls, img_chw, mask_hw, copy_cls)
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
                    ori_feature = self.prv_backbone_net(data_bchw)
                    ori_logit = self.prv_post_processor(ori_feature, ori_spatial_res, scale_factor=10)

                loss = self.criterion(output, target_bhw)

                # Feature extractor regularization + classifier regularization
                regularization_loss = l2_criterion(feature, ori_feature)
                regularization_loss = regularization_loss * self.cfg.TASK_SPECIFIC.GIFS.feature_reg_lambda # hyperparameter lambda
                loss = loss + regularization_loss
                # PD Loss from PIFS
                with torch.no_grad():
                    distill_output = post_processor_distillation_ref(ori_feature, ori_spatial_res, scale_factor=10)
                clf_loss = my_kd_criterion(output, distill_output) * self.cfg.TASK_SPECIFIC.GIFS.classifier_reg_lambda
                loss = loss + clf_loss

                optimizer.zero_grad() # reset gradient
                loss.backward()
                optimizer.step()
                scheduler.step()
                t.set_description_str("Loss: {:.4f}".format(loss.item()))