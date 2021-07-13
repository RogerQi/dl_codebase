import time
import numpy as np
import torch

import utils

from .trainer_base import trainer_base

class seg_trainer(trainer_base):
    def __init__(self, cfg, backbone_net, post_processor, criterion, device):
        self.cfg = cfg
        self.backbone_net = backbone_net
        self.post_processor = post_processor
        self.criterion = criterion
        self.device = device

    def train_one(self, device, train_loader, optimizer, epoch):
        self.backbone_net.train()
        self.post_processor.train()
        start_cp = time.time()
        train_total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
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
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), batch_acc, time.time() - start_cp))
        
        return train_total_loss / len(train_loader)

    def val_one(self, device, val_loader):
        self.backbone_net.eval()
        self.post_processor.eval()
        test_loss = 0
        correct = 0
        pixel_acc_list = []
        iou_list = []
        with torch.no_grad():
            for data, target in val_loader:
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
                        self.cfg.num_classes,
                        fg_only=self.cfg.METRIC.SEGMENTATION.fg_only
                    )
                    iou_list.append(float(iou))

            test_loss /= len(val_loader.dataset)

            m_iou = np.mean(iou_list)
            print('\nTest set: Average loss: {:.4f}, Mean Pixel Accuracy: {:.4f}, Mean IoU {:.4f}'.format(
                test_loss, np.mean(pixel_acc_list), m_iou))
        return m_iou

    def test_one(self, device, test_loader):
        return val_one(device, test_loader)
