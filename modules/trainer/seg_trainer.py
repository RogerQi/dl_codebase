import time
import numpy as np
import cv2
import torch
from tqdm import tqdm

import utils

from .trainer_base import trainer_base

scaler = torch.cuda.amp.GradScaler()

class seg_trainer(trainer_base):
    def __init__(self, cfg, backbone_net, post_processor, criterion, dataset_module, device):
        super(seg_trainer, self).__init__(cfg, backbone_net, post_processor, criterion, dataset_module, device)

    def train_one(self, device, optimizer, epoch):
        self.backbone_net.train()
        self.post_processor.train()
        start_cp = time.time()
        train_total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            optimizer.zero_grad() # reset gradient
            data, target = data.to(device), target.to(device)
            with torch.cuda.amp.autocast():
                feature = self.backbone_net(data)
                ori_spatial_res = data.shape[-2:]
                output = self.post_processor(feature, ori_spatial_res)
                loss = self.criterion(output, target)
            if True:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            train_total_loss += loss.item()
            if True:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if batch_idx % self.cfg.TRAIN.log_interval == 0:
                pred_map = output.max(dim = 1)[1]
                batch_acc, _ = utils.compute_pixel_acc(pred_map, target, fg_only=self.cfg.METRIC.SEGMENTATION.fg_only)
                print('Train Epoch: {0} [{1}/{2} ({3:.0f}%)]\tLoss: {4:.6f}\tBatch Pixel Acc: {5:.6f} Epoch Elapsed Time: {6:.1f}'.format(
                    epoch, batch_idx * len(data), len(self.train_set),
                    100. * batch_idx / len(self.train_loader), loss.item(), batch_acc, time.time() - start_cp))
        
        return train_total_loss / len(self.train_loader)

    def val_one(self, device):
        if self.cfg.meta_training_num_classes != -1:
            class_iou = self.eval_on_loader(self.val_loader, self.cfg.meta_training_num_classes)
        else:
            class_iou = self.eval_on_loader(self.val_loader, self.cfg.num_classes)
        print('Test set: Mean IoU {:.4f}'.format(np.mean(class_iou)))
        print("Class-wise IoU:")
        print(class_iou)
        return np.mean(class_iou)

    def test_one(self, device):
        return self.val_one(device)
    
    def eval_on_loader(self, test_loader, num_classes, visfreq=99999999):
        self.backbone_net.eval()
        self.post_processor.eval()
        test_loss = 0
        pixel_acc_list = []
        class_intersection, class_union = (None, None)
        class_names_list = test_loader.dataset.dataset.CLASS_NAMES_LIST
        with torch.no_grad():
            for idx, (data, target) in tqdm(enumerate(test_loader)):
                data, target = data.to(self.device), target.to(self.device)
                feature = self.backbone_net(data)
                ori_spatial_res = data.shape[-2:]
                output = self.post_processor(feature, ori_spatial_res)
                test_loss += self.criterion(output, target).item()  # sum up batch loss
                pred_map = output.max(dim = 1)[1]
                batch_acc, _ = utils.compute_pixel_acc(pred_map, target, fg_only=self.cfg.METRIC.SEGMENTATION.fg_only)
                pixel_acc_list.append(float(batch_acc))
                for i in range(pred_map.shape[0]):
                    pred_np = np.array(pred_map[i].cpu())
                    target_np = np.array(target[i].cpu(), dtype=np.int64)
                    intersection, union = utils.compute_iu(pred_np, target_np, num_classes)
                    if class_intersection is None:
                        class_intersection = intersection
                        class_union = union
                    else:
                        class_intersection += intersection
                        class_union += union
                    if (idx + 1) % visfreq == 0:
                        gt_label = utils.visualize_segmentation(self.cfg, data[i], target_np, class_names_list)
                        predicted_label = utils.visualize_segmentation(self.cfg, data[i], pred_np, class_names_list)
                        cv2.imwrite("{}_{}_pred.png".format(idx, i), predicted_label)
                        cv2.imwrite("{}_{}_label.png".format(idx, i), gt_label)
                        # Visualize RGB image as well
                        ori_rgb_np = np.array(data[i].permute((1, 2, 0)).cpu())
                        if 'normalize' in self.cfg.DATASET.TRANSFORM.TEST.transforms:
                            rgb_mean = self.cfg.DATASET.TRANSFORM.TEST.TRANSFORMS_DETAILS.NORMALIZE.mean
                            rgb_sd = self.cfg.DATASET.TRANSFORM.TEST.TRANSFORMS_DETAILS.NORMALIZE.sd
                            ori_rgb_np = (ori_rgb_np * rgb_sd) + rgb_mean
                        assert ori_rgb_np.max() <= 1.1, "Max is {}".format(ori_rgb_np.max())
                        ori_rgb_np[ori_rgb_np >= 1] = 1
                        ori_rgb_np = (ori_rgb_np * 255).astype(np.uint8)
                        # Convert to OpenCV BGR
                        ori_rgb_np = cv2.cvtColor(ori_rgb_np, cv2.COLOR_RGB2BGR)
                        cv2.imwrite("{}_{}_ori.jpg".format(idx, i), ori_rgb_np)

        test_loss /= len(test_loader.dataset)

        class_iou = class_intersection / (class_union + 1e-10)
        return class_iou
