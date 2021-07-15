import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm, trange

import classifier
import utils
from dataset.special_loader import get_fs_classification_loader

from .trainer_base import trainer_base

class fs_ft_clf_trainer(trainer_base):
    def __init__(self, cfg, backbone_net, post_processor, criterion, dataset_module, device):
        super(fs_ft_clf_trainer, self).__init__(cfg, backbone_net, post_processor, criterion, dataset_module, device)

        # init meta test loader
        self.meta_test_set = dataset_module.get_meta_test_set(cfg)

        n_iter = 600
        n_query = 15
        n_way = 5
        n_shot = 5
        self.meta_test_loader = get_fs_classification_loader(self.meta_test_set, n_iter, n_way, n_shot, n_query)

    def train_one(self, device, optimizer, epoch):
        self.backbone_net.train()
        self.post_processor.train()
        start_cp = time.time()
        train_total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            optimizer.zero_grad() # reset gradient
            data, target = data.to(device), target.to(device)
            feature = self.backbone_net(data)
            output = self.post_processor(feature)
            loss = self.criterion(output, target)
            loss.backward()
            train_total_loss += loss.item()
            optimizer.step()
            if batch_idx % self.cfg.TRAIN.log_interval == 0:
                pred = output.argmax(dim = 1, keepdim = True)
                correct_prediction = pred.eq(target.view_as(pred)).sum().item()
                batch_acc = correct_prediction / data.shape[0]
                print('Train Epoch: {0} [{1}/{2} ({3:.0f}%)]\tLoss: {4:.6f}\tBatch Acc: {5:.6f} Epoch Elapsed Time: {6:.1f}'.format(
                    epoch, batch_idx * len(data), len(self.train_set),
                    100. * batch_idx / len(self.train_loader), loss.item(), batch_acc, time.time() - start_cp))
        
        return train_total_loss / len(self.train_loader)

    def val_one(self, device):
        self.backbone_net.eval()
        self.post_processor.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(device), target.to(device)
                feature = self.backbone_net(data)
                output = self.post_processor(feature)
                test_loss += self.criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(dim = 1, keepdim = True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(self.val_set)

            acc = 100. * correct / len(self.val_set)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, len(self.val_set), acc))
        return acc

    def test_one(self, device):
        acc_list = []

        np.random.seed(1221)
        random.seed(1221)
        torch.manual_seed(1221)

        for j, meta_test_batch in tqdm(enumerate(self.meta_test_loader)):
            meta_test_batch = meta_test_batch[0]

            batch_acc = self.meta_test_single_pass(meta_test_batch)
            
            # Gather episode-wise statistics
            acc_list.append(batch_acc)

        # Gather test numbers
        print("Overall Acc Mean: {:.4f} Std: {:.4f} CI: {:.4f}".format(np.mean(acc_list), np.std(acc_list), 1.96 * np.std(acc_list) / np.sqrt(len(self.meta_test_loader))))

    def meta_test_single_pass(self, meta_test_batch):
        n_way = len(np.unique(meta_test_batch['supp_label_b']))

        temp_post_processor = classifier.dispatcher(self.cfg, self.feature_shape, n_way)
        temp_post_processor = temp_post_processor.to(self.device)

        query_img_bchw_tensor = meta_test_batch['query_img_bchw'].cuda()
        query_label_b_tensor = meta_test_batch['query_label_b'].cuda()
        supp_img_bchw_tensor = meta_test_batch['supp_img_bchw'].cuda()
        supp_label_b_tensor = meta_test_batch['supp_label_b'].cuda()

        # Fine-tune HEAD on few examples
        # optimizer parameters obtained from https://github.com/wyharveychen/CloserLookFewShot
        head_optimizer = torch.optim.SGD(temp_post_processor.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        support_size = supp_img_bchw_tensor.shape[0]
        batch_size = 4

        with torch.no_grad():
            all_features_bchw = self.backbone_net(supp_img_bchw_tensor)

        for epoch in range(100):
            # Evenly sample all samples
            permuted_idx = torch.randperm(support_size)
            for i in range(0, support_size, batch_size):
                with torch.no_grad():
                    selected_idx = permuted_idx[i:min(i + batch_size, support_size)]
                    batch_label_b = supp_label_b_tensor[selected_idx]
                    feature = all_features_bchw[selected_idx]
                soft_pred = temp_post_processor(feature, scale_factor=5)
                loss = self.criterion(soft_pred, batch_label_b)
                head_optimizer.zero_grad()
                loss.backward()
                head_optimizer.step()
        
        # Query set. Evaluation
        with torch.no_grad():
            query_feature = self.backbone_net(query_img_bchw_tensor)
            soft_pred = temp_post_processor(query_feature)
            pred = soft_pred.max(dim = 1)[1]
            acc = torch.mean((pred == query_label_b_tensor).float())
            return float(acc)