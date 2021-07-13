import __init_lib_path
from config_guard import cfg, update_config_from_yaml
import dataset
from dataset.mini_imagenet import get_bg_train_set
from dataset.special_loader import get_fs_clf_fg_bg_loader
import backbone
import classifier
import loss

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

from IPython import embed

def parse_args():
    parser = argparse.ArgumentParser(description = "Roger's Deep Learning Playground")
    parser.add_argument('--cfg', help = "specify particular yaml configuration to use", required = True,
        default = "configs/mnist_torch_official.taml", type = str)
    args = parser.parse_args()

    return args

def euclidean_dist( x, y):
    # x: (n_way * n_query, feature_len) = (n, d)
    # y: (n_way, feature_len) = (m, d)
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d) # (n, 1, d) => (n, m, d)
    y = y.unsqueeze(0).expand(n, m, d) # (1, m, d) => (n, m, d)

    return torch.pow(x - y, 2).sum(2)

def meta_train(cfg, backbone_net, criterion, device, optimizer, meta_train_set, bg_train_set, epoch):
    backbone_net.train()
    avg_loss = 0
    meta_loader = get_fs_clf_fg_bg_loader(meta_train_set, bg_train_set, 100, 5, 16)
    for j, meta_train_batch in enumerate(meta_loader):
        meta_train_batch = meta_train_batch[0]

        provided_labels = list(np.unique(meta_train_batch['supp_label_b']))
        n_way = len(provided_labels)
        assert n_way == 2
        assert provided_labels == [i for i in range(n_way)]

        query_img_bchw_tensor = meta_train_batch['query_img_bchw'].cuda()
        query_label_b_tensor = meta_train_batch['query_label_b'].cuda()
        supp_img_bchw_tensor = meta_train_batch['supp_img_bchw'].cuda()
        supp_label_b_tensor = meta_train_batch['supp_label_b'].cuda()

        support_size = supp_img_bchw_tensor.shape[0]

        supp_feature = backbone_net(supp_img_bchw_tensor)
        supp_proto = supp_feature.view((2, 5, 512)).mean(dim=1)

        query_feature = backbone_net(query_img_bchw_tensor)
        query_proto = query_feature.view((query_feature.shape[0], -1))
        query_dists = euclidean_dist(query_proto, supp_proto)
        query_scores = -query_dists
        loss = criterion(query_scores, query_label_b_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avg_loss += loss.item()
        if j % 20 == 0:
            print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, j, len(meta_loader), avg_loss/float(j+1)))

def main():
    # --------------------------
    # | Initial set up
    # |  1. Parse arg and yaml
    # |  2. Set device
    # |  3. Set seed
    # --------------------------
    args = parse_args()
    update_config_from_yaml(cfg, args)

    use_cuda = not cfg.SYSTEM.use_cpu
    device_str = "cuda" if use_cuda else "cpu"
    device = torch.device(device_str)

    kwargs = {'num_workers': cfg.SYSTEM.num_workers, 'pin_memory': cfg.SYSTEM.pin_memory} if use_cuda else {}

    torch.manual_seed(cfg.seed)

    # --------------------------
    # | Prepare datasets
    # --------------------------
    meta_train_set, meta_test_set = dataset.meta_dispatcher(cfg)
    bg_train_set = get_bg_train_set(cfg)

    print("Meta train set contains {} data points.".format(len(meta_train_set)))

    # 100 batches
    # each batch contains 5-way 5-shot. n_query=16

    # --------------------------
    # | Get ready to learn
    # |  1. Prepare network and loss
    # |  2. Prepare optimizer
    # |  3. Set learning rate
    # --------------------------
    backbone_net = backbone.dispatcher(cfg)
    backbone_net = backbone_net(cfg).to(device)
    feature_shape = backbone_net.get_feature_tensor_shape(device)
    print("Flatten feature length: {}".format(feature_shape))
    

    criterion = loss.dispatcher(cfg)

    trainable_params = backbone_net.parameters()

    if cfg.TRAIN.OPTIMIZER.type == "adadelta":
        optimizer = optim.Adadelta(trainable_params, lr = cfg.TRAIN.initial_lr,
                                    weight_decay = cfg.TRAIN.OPTIMIZER.weight_decay)
    elif cfg.TRAIN.OPTIMIZER.type == "SGD":
        optimizer = optim.SGD(trainable_params, lr = cfg.TRAIN.initial_lr, momentum = cfg.TRAIN.OPTIMIZER.momentum,
                                weight_decay = cfg.TRAIN.OPTIMIZER.weight_decay)
    elif cfg.TRAIN.OPTIMIZER.type == "ADAM":
        optimizer = optim.Adam(trainable_params, lr = cfg.TRAIN.initial_lr, betas = (0.9, 0.999),
                                weight_decay = cfg.TRAIN.OPTIMIZER.weight_decay)
    else:
        raise NotImplementedError("Got unsupported optimizer: {}".format(cfg.TRAIN.OPTIMIZER.type))

    # Prepare LR scheduler
    if cfg.TRAIN.lr_scheduler == "step_down":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = cfg.TRAIN.step_down_on_epoch,
                                                            gamma = cfg.TRAIN.step_down_gamma)
    elif cfg.TRAIN.lr_scheduler == "polynomial":
        max_epoch = cfg.TRAIN.max_epochs
        def polynomial_schedule(epoch):
            # from https://arxiv.org/pdf/2012.01415.pdf
            return (1 - epoch / max_epoch)**0.9
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, polynomial_schedule)
    else:
        raise NotImplementedError("Got unsupported scheduler: {}".format(cfg.TRAIN.lr_scheduler))

    best_val_metric = 0
    start_epoch = 1

    for epoch in range(start_epoch, cfg.TRAIN.max_epochs + 1):
        start_cp = time.time()
        meta_train(cfg, backbone_net, criterion, device, optimizer, meta_train_set, bg_train_set, epoch)
        scheduler.step()
        print("Training took {:.4f} seconds".format(time.time() - start_cp))
    
    if cfg.save_model:
        torch.save(
                {
                    "backbone": backbone_net.state_dict()
                },
                "{0}_final.pt".format(cfg.name)
            )

if __name__ == '__main__':
    main()
