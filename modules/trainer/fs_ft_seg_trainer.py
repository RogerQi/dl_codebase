import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

import classifier
import utils

from .trainer_base import trainer_base
from dataset.special_loader import get_fs_seg_loader

class fs_ft_seg_trainer(trainer_base):
    def __init__(self, cfg, backbone_net, post_processor, criterion, dataset_module, device):
        super(fs_ft_seg_trainer, self).__init__(cfg, backbone_net, post_processor, criterion, dataset_module, device)

        # init meta test loader
        self.meta_test_set = dataset_module.get_meta_test_set(cfg)

        n_iter = 1000
        n_query = 1
        n_way = 1
        n_shot = 1
        self.meta_test_loader = get_fs_seg_loader(self.meta_test_set, n_iter, n_way, n_shot, n_query)

    def train_one(self, device, optimizer, epoch):
        self.backbone_net.train()
        self.post_processor.train()
        start_cp = time.time()
        train_total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            optimizer.zero_grad() # reset gradient
            data, target = data.to(device), target.to(device)
            feature = self.backbone_net(data)
            ori_spatial_res = data.shape[-2:]
            output = self.post_processor(feature, ori_spatial_res)
            loss = self.criterion(output, target)
            loss.backward()
            train_total_loss += loss.item()
            optimizer.step()
            if batch_idx % self.cfg.TRAIN.log_interval == 0:
                pred_map = output.max(dim = 1)[1]
                batch_acc, _ = utils.compute_pixel_acc(pred_map, target, fg_only=self.cfg.METRIC.SEGMENTATION.fg_only)
                print('Train Epoch: {0} [{1}/{2} ({3:.0f}%)]\tLoss: {4:.6f}\tBatch Pixel Acc: {5:.6f} Epoch Elapsed Time: {6:.1f}'.format(
                    epoch, batch_idx * len(data), len(self.train_set),
                    100. * batch_idx / len(self.train_loader), loss.item(), batch_acc, time.time() - start_cp))
        
        return train_total_loss / len(self.train_loader)

    def val_one(self, device):
        self.backbone_net.eval()
        self.post_processor.eval()
        test_loss = 0
        correct = 0
        pixel_acc_list = []
        iou_list = []
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(device), target.to(device)
                feature = self.backbone_net(data)
                ori_spatial_res = data.shape[-2:]
                output = self.post_processor(feature, ori_spatial_res)
                test_loss += self.criterion(output, target).item()  # sum up batch loss
                pred_map = output.max(dim = 1)[1]
                batch_acc, _ = utils.compute_pixel_acc(pred_map, target, fg_only=self.cfg.METRIC.SEGMENTATION.fg_only)
                pixel_acc_list.append(float(batch_acc))
                for i in range(pred_map.shape[0]):
                    iou = utils.compute_iou(
                        np.array(pred_map[i].cpu()),
                        np.array(target[i].cpu(), dtype=np.int64),
                        self.cfg.meta_training_num_classes,
                        fg_only=self.cfg.METRIC.SEGMENTATION.fg_only
                    )
                    iou_list.append(float(iou))

            test_loss /= len(self.val_set)

            m_iou = np.mean(iou_list)
            print('\nTest set: Average loss: {:.4f}, Mean Pixel Accuracy: {:.4f}, Mean IoU {:.4f}'.format(
                test_loss, np.mean(pixel_acc_list), m_iou))
        return m_iou

    def test_one(self, device, num_runs=5):
        feature_shape = self.backbone_net.get_feature_tensor_shape(device)
        meta_test_task = [[None for j in range(1000)] for i in range(num_runs)]
        # Meta Test!
        iou_list = []
        for i in range(num_runs):
            for j, meta_test_batch in tqdm(enumerate(self.meta_test_loader)):
                meta_test_batch = meta_test_batch[0]

                tp_cnt, fp_cnt, fn_cnt, tn_cnt = meta_test_one(self.cfg, self.backbone_net, self.criterion, feature_shape, device, meta_test_batch)
                
                # Gather episode-wise statistics
                meta_test_task[i][j] = {}
                meta_test_task[i][j]['sampled_class_id'] = meta_test_batch['sampled_class_id']
                meta_test_task[i][j]['tp_pixel_cnt'] = tp_cnt
                meta_test_task[i][j]['fp_pixel_cnt'] = fp_cnt
                meta_test_task[i][j]['fn_pixel_cnt'] = fn_cnt
                meta_test_task[i][j]['tn_pixel_cnt'] = tn_cnt

            # Gather test numbers
            # TODO(roger): relax this to support dataset other than PASCAL-5i
            total_tp_cnt = np.zeros((5,), dtype = np.int)
            total_fp_cnt = np.zeros((5,), dtype = np.int)
            total_fn_cnt = np.zeros((5,), dtype = np.int)
            total_tn_cnt = np.zeros((5,), dtype = np.int)
            for j, meta_test_batch in enumerate(meta_test_task[i]):
                novel_class_id = meta_test_task[i][j]['sampled_class_id'] - 1 # from 1-indexed to 0-indexed
                total_tp_cnt[novel_class_id] += meta_test_task[i][j]['tp_pixel_cnt']
                total_fp_cnt[novel_class_id] += meta_test_task[i][j]['fp_pixel_cnt']
                total_fn_cnt[novel_class_id] += meta_test_task[i][j]['fn_pixel_cnt']
                total_tn_cnt[novel_class_id] += meta_test_task[i][j]['tn_pixel_cnt']
            single_run_acc = np.sum(total_tp_cnt + total_tn_cnt) / np.sum(total_tp_cnt + total_fp_cnt + total_fn_cnt + total_tn_cnt)
            classwise_iou = total_tp_cnt / (total_tp_cnt + total_fp_cnt + total_fn_cnt)
            single_run_iou = np.mean(classwise_iou)
            print("Accuracy: {:.4f} IoU: {:.4f}".format(single_run_acc, single_run_iou))
            iou_list.append(single_run_iou)
            
        print("Overall IoU Mean: {:.4f} Std: {:.4f}".format(np.mean(iou_list), np.std(iou_list)))

def masked_average_pooling(mask_b1hw, feature_bchw, normalization=False):
    '''
    Params
        - mask_b1hw: a binary mask whose element-wise value is either 0 or 1
        - feature_bchw: feature map obtained from the backbone
    
    Return: Mask-average-pooled vector of shape 1 x C
    '''
    if len(mask_b1hw.shape) == 3:
        mask_b1hw = mask_b1hw.view((mask_b1hw.shape[0], 1, mask_b1hw.shape[1], mask_b1hw.shape[2]))

    # Assert remove mask is not in mask provided
    assert -1 not in mask_b1hw

    # Spatial resolution mismatched. Interpolate feature to match mask size
    if mask_b1hw.shape[-2:] != feature_bchw.shape[-2:]:
        feature_bchw = F.interpolate(feature_bchw, size=mask_b1hw.shape[-2:], mode='bilinear')
    
    if normalization:
        feature_norm = torch.norm(feature_bchw, p=2, dim=1).unsqueeze(1).expand_as(feature_bchw)
        feature_bchw = feature_bchw.div(feature_norm + 1e-5) # avoid div by zero

    batch_pooled_vec = torch.sum(feature_bchw * mask_b1hw, dim = (2, 3)) / (mask_b1hw.sum(dim = (2, 3)) + 1e-5) # B x C
    return torch.mean(batch_pooled_vec, dim=0)

def meta_test_one(cfg, backbone_net, criterion, feature_shape, device, meta_test_batch):
    post_processor = classifier.dispatcher(cfg, feature_shape, 2)
    post_processor = post_processor.to(device)

    query_img_bchw_tensor = meta_test_batch['query_img_bchw'].cuda()
    query_mask_bhw_tensor = meta_test_batch['query_mask_bhw'].cuda()
    supp_img_bchw_tensor = meta_test_batch['supp_img_bchw'].cuda()
    supp_mask_bhw_tensor = meta_test_batch['supp_mask_bhw'].cuda()

    num_shots = supp_mask_bhw_tensor.shape[0]

    # Sanity check to make sure that there are at least some foreground pixels
    for b in range(supp_mask_bhw_tensor.shape[0]):
        assert 1 in supp_mask_bhw_tensor[b]
    for b in range(query_mask_bhw_tensor.shape[0]):
        assert 1 in query_mask_bhw_tensor[b]

    # Support set 1. Use masked average pooling to initialize class weight vector to bootstrap fine-tuning
    with torch.no_grad():
        support_feature = backbone_net(supp_img_bchw_tensor)
        fg_vec = masked_average_pooling(supp_mask_bhw_tensor == 1, support_feature, True)  # 1 x C
        bg_vec = masked_average_pooling(supp_mask_bhw_tensor == 0, support_feature, True)  # 1 x C
        fg_vec = fg_vec.reshape((1, -1, 1, 1)) # 1xCx1x1
        bg_vec = bg_vec.reshape((1, -1, 1, 1)) # 1xCx1x1
        bg_fg_class_mat = torch.cat([bg_vec, fg_vec], dim=0) #2xCx1x1
        post_processor.pixel_classifier.class_mat.weight.data = bg_fg_class_mat

    # Support set 2. TODO(roger): add back optimization to fine-tune initialized vectors

    # Query set. Evaluation
    with torch.no_grad():
        query_feature = backbone_net(query_img_bchw_tensor)
        eval_ori_spatial_res = query_img_bchw_tensor.shape[-2:]
        eval_predicted_mask = post_processor(query_feature, eval_ori_spatial_res)
        pred_map = eval_predicted_mask.max(dim = 1)[1]

        # TODO(roger): relax this to support multi-way
        # Following PANet, we use ignore_mask to mask confusing pixels from metric
        predicted_fg = torch.logical_and(pred_map == 1, query_mask_bhw_tensor != -1)
        predicted_bg = torch.logical_or(pred_map == 0, query_mask_bhw_tensor == -1)

        tp_cnt = torch.logical_and(predicted_fg, query_mask_bhw_tensor == 1).sum()
        fp_cnt = torch.logical_and(predicted_fg, query_mask_bhw_tensor != 1).sum()
        fn_cnt = torch.logical_and(predicted_bg, query_mask_bhw_tensor == 1).sum()
        tn_cnt = torch.logical_and(predicted_bg, query_mask_bhw_tensor != 1).sum()
    
    return tp_cnt, fp_cnt, fn_cnt, tn_cnt
