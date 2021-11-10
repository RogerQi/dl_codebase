import os
import random
import time
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
from torchvision import transforms
import heapq
from backbone.deeplabv3_renorm import BatchRenorm2d

import classifier
import utils

from .sequential_GIFS_seg_trainer import sequential_GIFS_seg_trainer
from IPython import embed

class scene_clf_head(nn.Module):
    def __init__(self, indim, outdim):
        super(scene_clf_head, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.intermediate = nn.Linear(indim, 512, bias=False)
        self.L = nn.Linear( 512, outdim, bias = False)  

        self.scale_factor = 30

    def forward(self, x):
        x_normalized = self.get_cos_embedding(x)
        L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
        self.L.weight.data = self.L.weight.data.div(L_norm + 1e-5)
        cos_dist = self.L(x_normalized)

        return self.scale_factor * cos_dist
    
    def get_cos_embedding(self, x):
        x = self.avgpool(x)
        assert len(x.shape) == 4 # BCHW
        x = torch.flatten(x, start_dim = 1)
        x = self.intermediate(x)
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x = x.div(x_norm+ 1e-5)
        return x

def harmonic_mean(base_iou, novel_iou):
    return 2 / (1. / base_iou + 1. / novel_iou)

memory_bank_size = 500

class fs_incremental_trainer(sequential_GIFS_seg_trainer):
    def __init__(self, cfg, backbone_net, post_processor, criterion, dataset_module, device):
        super(fs_incremental_trainer, self).__init__(cfg, backbone_net, post_processor, criterion, dataset_module, device)
        
        self.partial_data_pool = {}
        self.blank_bank = {}
        self.demo_pool = {}

        self.train_set_vanilla_label = dataset_module.get_train_set_vanilla_label(cfg)

        # init a scene classification head
        self.scene_classifier = scene_clf_head(2048, 365).to(self.device) # 365 classes

        self.scene_classifier_trained = False
        assert self.cfg.TASK_SPECIFIC.GIFS.synthetic_blending in ('none', 'harmonization', 'gaussian')
        self.loaded_weight_path = None

        if self.cfg.TASK_SPECIFIC.GIFS.synthetic_blending == 'harmonization':
            # Image harmonization
            # Credit: https://arxiv.org/abs/1911.13239
            torchscript_path = '/data/pretrained_model/DoveNet.pt'
            self.netG = torch.jit.load(torchscript_path)
            self.netG = self.netG.to(torch.device('cuda'))
            self.netG.eval()
    
    def construct_baseset(self):
        baseset_type = self.cfg.TASK_SPECIFIC.GIFS.baseset_type
        baseset_folder = f"save_{self.cfg.name}"
        if self.cfg.TASK_SPECIFIC.GIFS.load_baseset:
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
        self.base_img_candidates = self.construct_baseset()
        if self.cfg.TASK_SPECIFIC.GIFS.context_aware_sampling:
            self.scene_model_setup()
        sequential_GIFS_seg_trainer.test_one(self, device, num_runs)
    
    def synthesizer_sample(self, novel_obj_id):
        num_existing_objects = num_novel_objects = 2
        # Sample an image from base memory bank
        if self.cfg.TASK_SPECIFIC.GIFS.context_aware_sampling:
            if torch.rand(1) < 0.5:
                base_img_idx = np.random.choice(self.base_data_w_context)
            else:
                base_img_idx = np.random.choice(self.base_data_no_context)
        else:
            # Sample from complete data pool (base dataset)
            base_img_idx = np.random.choice(self.base_img_candidates)
        syn_img_chw, syn_mask_hw = self.train_set_vanilla_label[base_img_idx]
        # Sample from partial data pool
        # Synthesis probabilities are computed using virtual RFS
        # TODO(roger): automate this probability computation
        if len(self.partial_data_pool) > 1 and torch.rand(1) < 0.5568: # VOC: 0.5568 COCO: 0.5124
            # select an old class
            candidate_classes = [c for c in self.partial_data_pool.keys() if c != novel_obj_id]
            for i in range(num_existing_objects):
                selected_class = np.random.choice(candidate_classes)
                selected_sample = random.choice(self.partial_data_pool[selected_class])
                img_chw, mask_hw = selected_sample
                syn_img_chw, syn_mask_hw = self.copy_and_paste(img_chw, mask_hw, syn_img_chw, syn_mask_hw, selected_class)

        if torch.rand(1) < 0.7424: # VOC: 0.7424 COCO: 0.6832
            for i in range(num_novel_objects):
                selected_sample = random.choice(self.partial_data_pool[novel_obj_id])
                img_chw, mask_hw = selected_sample
                syn_img_chw, syn_mask_hw = self.copy_and_paste(img_chw, mask_hw, syn_img_chw, syn_mask_hw, novel_obj_id)

        return (syn_img_chw, syn_mask_hw)
    
    def init_scene_training(self):
        # init scene dataset and loader. The default loader is segmentation dataset
        import dataset.places365_stanford as places365
        self.scene_train_set = places365.Places365StanfordReader(utils.get_dataset_root(), False)
        self.scene_train_set = utils.dataset_normalization_wrapper(self.cfg, self.scene_train_set)
        self.scene_train_loader = torch.utils.data.DataLoader(self.scene_train_set, batch_size=16, shuffle=True, **self.loader_kwargs)
        # define my own optimizer
        self.scene_optimizer = optim.SGD(self.scene_classifier.parameters(), lr=1e-2)
        # define my own loss
        self.scene_criterion = nn.CrossEntropyLoss(ignore_index=-1).to(self.device)
    
    def scene_train_one(self, epoch):
        self.backbone_net.eval()
        self.post_processor.eval()
        self.scene_classifier.train()
        start_cp = time.time()
        train_total_loss = 0
        total_correct = 0
        total_size = 0
        for batch_idx, (data, target) in enumerate(self.scene_train_loader):
            self.scene_optimizer.zero_grad() # reset gradient
            data, target = data.to(self.device), target.to(self.device)
            with torch.no_grad():
                feature_map = self.backbone_net.feature_forward(data)
            output = self.scene_classifier(feature_map)
            loss = self.scene_criterion(output, target)
            loss.backward()
            train_total_loss += loss.item()
            self.scene_optimizer.step()
            if batch_idx % 200 == 0:
                pred = output.argmax(dim = 1, keepdim = True)
                correct_prediction = pred.eq(target.view_as(pred)).sum().item()
                batch_acc = correct_prediction / data.shape[0]
                total_correct += correct_prediction
                total_size += data.shape[0]
                print('Train Epoch: {0} [{1}/{2} ({3:.0f}%)]\tLoss: {4:.6f}\tBatch Acc: {5:.6f} Epoch Elapsed Time: {6:.1f}'.format(
                    epoch, batch_idx * len(data), len(self.scene_train_set),
                    100. * batch_idx / len(self.scene_train_loader), loss.item(), batch_acc, time.time() - start_cp))
        
        return train_total_loss / len(self.train_loader), total_correct / total_size
    
    def train_scene_model(self, max_epoch=10):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.scene_optimizer, max_epoch)
        for epoch in range(1, max_epoch):
            start_cp = time.time()
            train_loss, train_acc = self.scene_train_one(epoch)
            scheduler.step()
            print("Training took {:.4f} seconds".format(time.time() - start_cp))
            print("Training acc: {:.4f} Training loss: {:.4f}".format(train_acc, train_loss))
            print("===================================\n")
    
    def save_model(self, file_path):
        """Save default model (backbone_net, post_processor to a specified file path)

        Args:
            file_path (str): path to save the model
        """
        if self.scene_classifier_trained:
            torch.save({
                "backbone": self.backbone_net.state_dict(),
                "head": self.post_processor.state_dict(),
                "scene_clf_head": self.scene_classifier.state_dict(),
            }, file_path)
        else:
            torch.save({
                "backbone": self.backbone_net.state_dict(),
                "head": self.post_processor.state_dict()
            }, file_path)
    
    def load_model(self, file_path):
        """Load weights for default model components (backbone_net, post_process) from a given file path

        Args:
            file_path (str): path to trained weights
        """
        trained_weight_dict = torch.load(file_path, map_location=self.device)
        self.backbone_net.load_state_dict(trained_weight_dict['backbone'], strict=True)
        self.post_processor.load_state_dict(trained_weight_dict['head'], strict=True)
        self.loaded_weight_path = file_path
        if 'scene_clf_head' in trained_weight_dict.keys():
            self.scene_classifier.load_state_dict(trained_weight_dict['scene_clf_head'], strict=True)
            self.scene_classifier_trained = True
    
    def scene_model_setup(self):
        # if option is selected and no existing model is available, train
        if not self.scene_classifier_trained:
            self.init_scene_training()
            self.train_scene_model()
            self.scene_classifier_trained = True
            self.save_model(self.loaded_weight_path)
        # Compute feature vectors for data in the pool
        self.base_pool_cos_embeddings = []
        for base_data_idx in self.base_img_candidates:
            img_chw, _ = self.train_set_vanilla_label[base_data_idx]
            img_bchw = img_chw.view((1,) + img_chw.shape).to(self.device)
            with torch.no_grad():
                feature_map = self.backbone_net.feature_forward(img_bchw)
                cos_vec_bc = self.scene_classifier.get_cos_embedding(feature_map) # B x C (1 x 512)
                self.base_pool_cos_embeddings.append(cos_vec_bc.squeeze()) # 512
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

        if self.cfg.TASK_SPECIFIC.GIFS.synthetic_blending == 'harmonization':
            img_chw = self.harmonize_image(img_chw, (mask_hw == mask_id))

        return (img_chw, mask_hw)
    
    def harmonize_image(self, image_tensor, mask_tensor):
        # Unnormalize image to torch.float32 between 0-1
        rgb_mean = self.cfg.DATASET.TRANSFORM.TEST.TRANSFORMS_DETAILS.NORMALIZE.mean
        rgb_sd = self.cfg.DATASET.TRANSFORM.TEST.TRANSFORMS_DETAILS.NORMALIZE.sd
        rgb_mean = torch.tensor(rgb_mean).view((3, 1, 1))
        rgb_sd = torch.tensor(rgb_sd).view((3, 1, 1))
        image_tensor = (image_tensor * rgb_sd) + rgb_mean
        image_tensor[image_tensor < 0] = 0
        image_tensor[image_tensor > 1] = 1
        # Normalize image using 0.5 mean and 0.5 std
        normalize_func = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image_tensor = normalize_func(image_tensor) # CHW
        # Mask is binary in 0/1 but is float
        mask_tensor = mask_tensor.to(torch.float32)
        mask_tensor = mask_tensor.view((1,) + mask_tensor.shape) # 1HW
        # Pad input
        inputs = torch.cat([image_tensor, mask_tensor]) # 4HW
        inputs = inputs.view((1,) + inputs.shape) # 14HW
        # Harmonize
        with torch.no_grad():
            output = self.netG(inputs.cuda()) # 13HW, -1~1
        # Normalize it back to 0-1 and then to custom mean sd...
        output = (output + 1) / 2
        my_normalize = transforms.Normalize(mean=self.cfg.DATASET.TRANSFORM.TEST.TRANSFORMS_DETAILS.NORMALIZE.mean,
                                    std=self.cfg.DATASET.TRANSFORM.TEST.TRANSFORMS_DETAILS.NORMALIZE.sd)
        output = my_normalize(output)
        assert output.shape[0] == 1
        return output[0].cpu()

    def finetune_backbone(self, base_class_idx, novel_class_idx, supp_img_bchw, supp_mask_bhw):
        assert self.prv_backbone_net is not None
        assert self.prv_post_processor is not None

        if self.cfg.TASK_SPECIFIC.GIFS.context_aware_sampling:
            self.base_data_w_context = []

        for b in range(supp_img_bchw.shape[0]):
            novel_img_chw = supp_img_bchw[b]

            # Compute cosine embedding
            if self.cfg.TASK_SPECIFIC.GIFS.context_aware_sampling:
                novel_img_bchw = novel_img_chw.view((1,) + novel_img_chw.shape).to(self.device)
                with torch.no_grad():
                    feature_map = self.vanilla_backbone_net.feature_forward(novel_img_bchw)
                    cos_vec_bc = self.scene_classifier.get_cos_embedding(feature_map) # B x C (1 x 512)
                    similarity_score = F.cosine_similarity(cos_vec_bc, self.base_pool_cos_embeddings)
                    base_candidates = torch.argsort(similarity_score)[-int(0.1 * self.base_pool_cos_embeddings.shape[0]):] # B integer array
                    self.base_data_w_context += list(base_candidates.cpu().numpy())

            mask_hw = supp_mask_bhw[b]
            for novel_obj_id in novel_class_idx:
                novel_mask_hw = (mask_hw == novel_obj_id)

                novel_mask_hw_np = novel_mask_hw.numpy().astype(np.uint8)

                # RETR_EXTERNAL to keep online the outer contour
                contours, _ = cv2.findContours(novel_mask_hw_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if True:
                    # whole image
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
                    if novel_obj_id not in self.partial_data_pool:
                        self.partial_data_pool[novel_obj_id] = []
                    mask_roi = novel_mask_hw[y_min:y_max,x_min:x_max]
                    img_roi = novel_img_chw[:,y_min:y_max,x_min:x_max]
                    self.partial_data_pool[novel_obj_id].append((img_roi, mask_roi))
                else:
                    # Register new masks by parts
                    for ctr in contours:
                        x, y, w, h = cv2.boundingRect(ctr)
                        if novel_obj_id not in self.partial_data_pool:
                            self.partial_data_pool[novel_obj_id] = []
                        mask_roi = novel_mask_hw[y:y+h,x:x+w]
                        if torch.sum(mask_roi) < 100:
                            continue
                        img_roi = novel_img_chw[:,y:y+h,x:x+w]
                        self.partial_data_pool[novel_obj_id].append((img_roi, mask_roi))

        if self.cfg.TASK_SPECIFIC.GIFS.context_aware_sampling:
            self.base_data_w_context = list(set(self.base_data_w_context))
            self.base_data_no_context = [i for i in range(memory_bank_size) if i not in self.base_data_w_context]

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
                    novel_obj_id = random.choice(novel_class_idx)
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
                    # self.vanilla_backbone_net for the base version
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
                if False:
                    # regularization on weights itself
                    vanilla_classifier_weights = self.vanilla_post_processor.pixel_classifier.class_mat.weight.data
                    new_classifier_weights = self.post_processor.pixel_classifier.class_mat.weight.data[self.vanilla_base_class_idx]
                    clf_loss = l2_criterion(new_classifier_weights, vanilla_classifier_weights) * self.cfg.TASK_SPECIFIC.GIFS.classifier_reg_lambda
                else:
                    # regularization on output logits
                    clf_loss = l2_criterion(output[:,base_class_idx,:,:], ori_logit) * self.cfg.TASK_SPECIFIC.GIFS.classifier_reg_lambda
                loss = loss + clf_loss

                optimizer.zero_grad() # reset gradient
                loss.backward()
                optimizer.step()
                scheduler.step()
                t.set_description_str("Loss: {:.4f}".format(loss.item()))
