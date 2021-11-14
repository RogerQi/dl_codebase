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
import torchvision
import torchvision.transforms.functional as tr_F
from tqdm import tqdm, trange
from torchvision import transforms
import heapq
from backbone.deeplabv3_renorm import BatchRenorm2d

import classifier
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
    
    def construct_baseset(self):
        baseset_type = self.cfg.TASK_SPECIFIC.GIFS.baseset_type
        baseset_folder = f"save_{self.cfg.name}"
        if self.cfg.TASK_SPECIFIC.GIFS.load_baseset and baseset_type != 'random':
            print(f"load baseset from {baseset_folder}/examplar_list_{baseset_type}")
            examplar_list = torch.load(f"{baseset_folder}/examplar_list_{baseset_type}")
        elif baseset_type == 'random':
            print(f"construct {baseset_type} baseset for {self.cfg.name}")
            examplar_list = np.arange(0, len(self.train_set_vanilla_label))
        elif baseset_type in ['far', 'close', 'far_close']:
            print(f"construct {baseset_type} baseset for {self.cfg.name}")
            self.prv_backbone_net = deepcopy(self.backbone_net)
            self.prv_post_processor = deepcopy(self.post_processor)
            
            self.prv_backbone_net.eval()
            self.prv_post_processor.eval()

            base_id_list = self.train_set_vanilla_label.dataset.get_label_range()

            print(f"self.base_id_list contains {base_id_list}")
            base_id_set = set(base_id_list)

            m = (memory_bank_size // len(base_id_list)) * 2
            if 'far' in baseset_type:
                m_far = m
                similarity_farthest_dic = {}
                k_far = {}
            if 'close' in baseset_type:
                m_close = m
                similarity_closest_dic = {}
                k_close = {}
            if baseset_type == 'far_close':
                m_close //= 2
                m_far //= 2
            mean_weight_dic = {}

            # Get the mean weight of each class
            for c in base_id_list:
                mean_weight_dic[c] = self.prv_post_processor.pixel_classifier.class_mat.weight.data[c]
                mean_weight_dic[c] = mean_weight_dic[c].reshape((-1))
                mean_weight_dic[c] = mean_weight_dic[c].cpu().unsqueeze(0)

            for c in base_id_set:
                if 'far' in baseset_type:
                    k_far[c] = min(m_far, len(self.train_set_vanilla_label.dataset.get_class_map(c)))
                    similarity_farthest_dic[c] = []
                if 'close' in baseset_type:
                    k_close[c] = min(m_close, len(self.train_set_vanilla_label.dataset.get_class_map(c)))
                    similarity_closest_dic[c] = []
                    
            # Maintain a m-size heap to store the top m images of each class
            for i in tqdm(range(len(self.train_set_vanilla_label))):
                img, mask = self.train_set_vanilla_label[i]
                class_list = torch.unique(mask).tolist()
                img_tensor = torch.stack([img]).cuda()
                mask_tensor = torch.stack([mask]).cuda()
                for c in class_list:
                    if c not in base_id_set:
                        continue
                    with torch.no_grad():
                        img_feature = self.prv_backbone_net(img_tensor)
                        img_weight = utils.masked_average_pooling(mask_tensor == c, img_feature, True)
                    img_weight = img_weight.cpu().unsqueeze(0)
                    similarity = F.cosine_similarity(img_weight, mean_weight_dic[c])
                    # Update closest heap
                    if 'close' in baseset_type:
                        if len(similarity_closest_dic[c]) < k_close[c]:
                            heapq.heappush(similarity_closest_dic[c], ((similarity, i)))
                        else:
                            heapq.heappushpop(similarity_closest_dic[c], ((similarity, i)))
                    # Update farthest heap
                    if 'far' in baseset_type:
                        if len(similarity_farthest_dic[c]) < k_far[c]:
                            heapq.heappush(similarity_farthest_dic[c], ((-similarity, i)))
                        else:
                            heapq.heappushpop(similarity_farthest_dic[c], ((-similarity, i)))   
            if not os.path.exists(baseset_folder):   
                os.makedirs(baseset_folder)  
            if 'close' in baseset_type:
                torch.save(similarity_closest_dic, f"{baseset_folder}/similarity_closest_dic_{baseset_type}")
            if 'far' in baseset_type:
                torch.save(similarity_farthest_dic, f"{baseset_folder}/similarity_farthest_dic_{baseset_type}")

            # Combine all the top images of each class by set union
            examplar_set = set()
            for c in base_id_list:
                if 'close' in baseset_type:
                    close_list = similarity_closest_dic[c]
                    class_examplar_list = [i for similarity, i in close_list]
                    examplar_set = examplar_set.union(class_examplar_list)

                if 'far' in baseset_type:
                    far_list = similarity_farthest_dic[c]
                    class_examplar_list = [i for similarity, i in far_list]
                    examplar_set = examplar_set.union(class_examplar_list)

            examplar_list = sorted(list(examplar_set))
            torch.save(examplar_list, f"{baseset_folder}/examplar_list_{baseset_type}")
        else:
            raise AssertionError('invalid baseset_type', baseset_type)
            
        print(f"total number of examplar_list {len(examplar_list)}")
        return np.random.choice(examplar_list, replace=False, size=(memory_bank_size,))
    
    def test_one(self, device, num_runs=5):
        if self.cfg.TASK_SPECIFIC.GIFS.num_runs != -1:
            num_runs = self.cfg.TASK_SPECIFIC.GIFS.num_runs
        self.base_img_candidates = self.construct_baseset()
        if self.cfg.TASK_SPECIFIC.GIFS.context_aware_sampling:
            self.scene_model_setup()
        sequential_GIFS_seg_trainer.test_one(self, device, num_runs)
    
    def synthesizer_sample(self, novel_obj_id):
        num_existing_objects = 2
        num_novel_objects = 2
        # Sample an image from base memory bank
        if self.cfg.TASK_SPECIFIC.GIFS.context_aware_sampling:
            if torch.rand(1) < 0.5:
                memory_buffer_idx = np.random.choice(self.context_similar_map[novel_obj_id])
                base_img_idx = self.base_img_candidates[memory_buffer_idx]
            else:
                # Complement of self.context_similar_map[novel_obj_id]
                unrelated_idx_list = [i for i in range(memory_bank_size) if i not in self.context_similar_map[novel_obj_id]]
                memory_buffer_idx = np.random.choice(unrelated_idx_list)
                base_img_idx = self.base_img_candidates[memory_buffer_idx]
        else:
            # Sample from complete data pool (base dataset)
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
            r_n = r_e * 2
            other_prob = (r_e + r_n) / (r_e + r_n + 1 + r_n)
            selected_novel_prob = (r_n + r_n) / (r_e + r_n + 1 + r_n)
        elif self.cfg.TASK_SPECIFIC.GIFS.probabilistic_synthesis_strat == 'always':
            other_prob = 0
            selected_novel_prob = 1
        elif self.cfg.TASK_SPECIFIC.GIFS.probabilistic_synthesis_strat == 'always_no':
            other_prob = 0
            selected_novel_prob = 0
        elif self.cfg.TASK_SPECIFIC.GIFS.probabilistic_synthesis_strat == 'CAS':
            other_prob = 1. / total_classes
            selected_novel_prob = 1. / total_classes
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
    
    def load_model(self, file_path):
        """Load weights for default model components (backbone_net, post_process) from a given file path

        Args:
            file_path (str): path to trained weights
        """
        trained_weight_dict = torch.load(file_path, map_location=self.device)
        self.backbone_net.load_state_dict(trained_weight_dict['backbone'], strict=True)
        self.post_processor.load_state_dict(trained_weight_dict['head'], strict=True)
    
    def get_scene_embedding(self, img):
        '''
        img: normalized image tensor of shape CHW
        '''
        assert len(img.shape) == 3 # CHW
        assert img.shape[1] == img.shape[2]
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
        self.scene_model = self.scene_model.cuda()
        # Compute feature vectors for data in the pool
        self.base_pool_cos_embeddings = []
        for base_data_idx in self.base_img_candidates:
            img_chw, _ = self.train_set_vanilla_label[base_data_idx]
            img_chw = img_chw.to(self.device)
            scene_embedding = self.get_scene_embedding(img_chw)
            assert len(scene_embedding.shape) == 1
            self.base_pool_cos_embeddings.append(scene_embedding)
        self.base_pool_cos_embeddings = torch.stack(self.base_pool_cos_embeddings)
    
    def copy_and_paste(self, novel_img_chw, novel_mask_hw, base_img_chw, base_mask_hw, mask_id):
        # Horizontal Flipping
        if torch.rand(1) < 0.5:
            novel_img_chw = tr_F.hflip(novel_img_chw)
            novel_mask_hw = tr_F.hflip(novel_mask_hw)

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

    def finetune_backbone(self, base_class_idx, novel_class_idx, supp_img_bchw, supp_mask_bhw):
        assert self.prv_backbone_net is not None
        assert self.prv_post_processor is not None

        if self.cfg.TASK_SPECIFIC.GIFS.context_aware_sampling:
            self.context_similar_map = {}

        for b in range(supp_img_bchw.shape[0]):
            novel_img_chw = supp_img_bchw[b]
            mask_hw = supp_mask_bhw[b]
            
            # Each image contains only 1 novel class (to ensure only num_shots samples are presented)
            for novel_obj_id in novel_class_idx:
                if novel_obj_id in mask_hw:
                    break
            
            # Sanity check to ensure above statement
            for class_id in novel_class_idx:
                if class_id == novel_obj_id: continue
                assert class_id not in mask_hw
            
            # Compute cosine embedding
            if self.cfg.TASK_SPECIFIC.GIFS.context_aware_sampling:
                scene_embedding = self.get_scene_embedding(novel_img_chw.cuda())
                scene_embedding = scene_embedding.view((1,) + scene_embedding.shape)
                similarity_score = F.cosine_similarity(scene_embedding, self.base_pool_cos_embeddings)
                base_candidates = torch.argsort(similarity_score)[-int(0.1 * self.base_pool_cos_embeddings.shape[0]):] # Indices array
                if novel_obj_id not in self.context_similar_map:
                    self.context_similar_map[novel_obj_id] = list(base_candidates.cpu().numpy())
                else:
                    self.context_similar_map[novel_obj_id] += list(base_candidates.cpu().numpy())

            novel_mask_hw = (mask_hw == novel_obj_id)

            novel_mask_hw_np = novel_mask_hw.numpy().astype(np.uint8)

            # RETR_EXTERNAL to keep online the outer contour
            contours, _ = cv2.findContours(novel_mask_hw_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Crop annotated objects off the image
            # Compute a minimum rectangle containing the object
            if len(contours) == 0: continue
            cnt = contours[0]
            x_min = tuple(cnt[cnt[:,:,0].argmin()][0])[0]
            x_max = tuple(cnt[cnt[:,:,0].argmax()][0])[0]
            y_min = tuple(cnt[cnt[:,:,1].argmin()][0])[1]
            y_max = tuple(cnt[cnt[:,:,1].argmax()][0])[1]
            for cnt in contours:
                x_min = min(x_min, tuple(cnt[cnt[:,:,0].argmin()][0])[0])
                x_max = max(x_max, tuple(cnt[cnt[:,:,0].argmax()][0])[0])
                y_min = min(y_min, tuple(cnt[cnt[:,:,1].argmin()][0])[1])
                y_max = max(y_max, tuple(cnt[cnt[:,:,1].argmax()][0])[1])
            # Minimum bounding rectangle computed; now register it to the data pool
            if novel_obj_id not in self.partial_data_pool:
                self.partial_data_pool[novel_obj_id] = []
            # mask_roi is a boolean array
            mask_roi = novel_mask_hw[y_min:y_max,x_min:x_max]
            img_roi = novel_img_chw[:,y_min:y_max,x_min:x_max]
            self.partial_data_pool[novel_obj_id].append((img_roi, mask_roi))

        if self.cfg.TASK_SPECIFIC.GIFS.context_aware_sampling:
            for c in self.context_similar_map:
                self.context_similar_map[c] = list(set(self.context_similar_map[c]))

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
                        idx = np.random.randint(supp_img_bchw.shape[0])
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
                    # self.vanilla_backbone_net for the base version
                    ori_feature = self.prv_backbone_net(data_bchw)
                    ori_logit = self.prv_post_processor(ori_feature, ori_spatial_res, scale_factor=10)

                # if self.cfg.TASK_SPECIFIC.GIFS.pseudo_base_label:
                #     novel_mask = torch.zeros_like(target_bhw)
                #     for novel_idx in novel_class_idx:
                #         novel_mask = torch.logical_or(novel_mask, target_bhw == novel_idx)
                #     tmp_target_bhw = output.max(dim = 1)[1]
                #     tmp_target_bhw[novel_mask] = target_bhw[novel_mask]
                #     target_bhw = tmp_target_bhw

                loss = self.criterion(output, target_bhw)

                # Feature extractor regularization + classifier regularization
                regularization_loss = l2_criterion(feature, ori_feature)
                regularization_loss = regularization_loss * self.cfg.TASK_SPECIFIC.GIFS.feature_reg_lambda # hyperparameter lambda
                loss = loss + regularization_loss
                # L2 regulalrization on base classes
                # TODO: to be removed
                if True:
                    with torch.no_grad():
                        distill_output = post_processor_distillation_ref(ori_feature, ori_spatial_res, scale_factor=10)
                    clf_loss = my_kd_criterion(output, distill_output) * self.cfg.TASK_SPECIFIC.GIFS.classifier_reg_lambda
                    # regularization on weights itself
                    # vanilla_classifier_weights = self.vanilla_post_processor.pixel_classifier.class_mat.weight.data
                    # new_classifier_weights = self.post_processor.pixel_classifier.class_mat.weight.data[self.vanilla_base_class_idx]
                    # clf_loss = l2_criterion(new_classifier_weights, vanilla_classifier_weights) * self.cfg.TASK_SPECIFIC.GIFS.classifier_reg_lambda
                else:
                    # regularization on output logits
                    clf_loss = l2_criterion(output[:,base_class_idx,:,:], ori_logit) * self.cfg.TASK_SPECIFIC.GIFS.classifier_reg_lambda
                loss = loss + clf_loss

                optimizer.zero_grad() # reset gradient
                loss.backward()
                optimizer.step()
                scheduler.step()
                t.set_description_str("Loss: {:.4f}".format(loss.item()))
