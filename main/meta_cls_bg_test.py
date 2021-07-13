import __init_lib_path
from config_guard import cfg, update_config_from_yaml
import dataset
from dataset.mini_imagenet import get_bg_test_set
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
    parser.add_argument('--load', help="specify saved checkpoint to evaluate", required=True, type=str)
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

def meta_test_one_proto(cfg, backbone_net, criterion, feature_shape, device, meta_test_batch):
    n_way = len(np.unique(meta_test_batch['supp_label_b']))
    assert n_way == 2, "only FG and BG classes"

    provided_labels = list(np.unique(meta_test_batch['supp_label_b']))
    n_way = len(provided_labels)
    assert provided_labels == [i for i in range(n_way)]

    query_img_bchw_tensor = meta_test_batch['query_img_bchw'].cuda()
    query_label_b_tensor = meta_test_batch['query_label_b'].cuda()
    supp_img_bchw_tensor = meta_test_batch['supp_img_bchw'].cuda()
    supp_label_b_tensor = meta_test_batch['supp_label_b'].cuda()

    with torch.no_grad():
        supp_feature = backbone_net(supp_img_bchw_tensor)
        supp_proto = supp_feature.view((2, 5, 512)).mean(dim=1)

        query_feature = backbone_net(query_img_bchw_tensor)
        query_proto = query_feature.view((query_feature.shape[0], -1))
        query_dists = euclidean_dist(query_proto, supp_proto)
        query_scores = -query_dists
        pred = query_scores.max(dim=1)[1].cpu().numpy()
        label = query_label_b_tensor.cpu().numpy()
    
    tp = np.sum(np.logical_and(pred == 1, label == 1))
    tn = np.sum(np.logical_and(pred == 0, label == 0))
    fp = np.sum(np.logical_and(pred == 1, label == 0))
    fn = np.sum(np.logical_and(pred == 0, label == 1))

    return tp, tn, fp, fn

def meta_test_one(cfg, backbone_net, criterion, feature_shape, device, meta_test_batch):
    n_way = len(np.unique(meta_test_batch['supp_label_b']))
    assert n_way == 2, "only FG and BG classes"

    post_processor = classifier.dispatcher(cfg, feature_shape, n_way)
    post_processor = post_processor.to(device)

    query_img_bchw_tensor = meta_test_batch['query_img_bchw'].cuda()
    query_label_b_tensor = meta_test_batch['query_label_b'].cuda()
    supp_img_bchw_tensor = meta_test_batch['supp_img_bchw'].cuda()
    supp_label_b_tensor = meta_test_batch['supp_label_b'].cuda()

    # Fine-tune HEAD on few examples
    head_optimizer = torch.optim.SGD(post_processor.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

    support_size = supp_img_bchw_tensor.shape[0]
    batch_size = 4

    with torch.no_grad():
        all_features_bchw = backbone_net(supp_img_bchw_tensor)

    for epoch in range(100):
        # Evenly sample all samples
        permuted_idx = torch.randperm(support_size)
        for i in range(0, support_size, batch_size):
            with torch.no_grad():
                selected_idx = permuted_idx[i:min(i + batch_size, support_size)]
                batch_label_b = supp_label_b_tensor[selected_idx]
                feature = all_features_bchw[selected_idx]
            soft_pred = post_processor(feature, scale_factor=5)
            loss = criterion(soft_pred, batch_label_b)
            head_optimizer.zero_grad()
            loss.backward()
            head_optimizer.step()
    
    # Query set. Evaluation
    with torch.no_grad():
        query_feature = backbone_net(query_img_bchw_tensor)
        soft_pred = post_processor(query_feature)
        pred = soft_pred.max(dim = 1)[1].cpu().numpy()
        label = query_label_b_tensor.cpu().numpy()
    tp = np.sum(np.logical_and(pred == 1, label == 1))
    tn = np.sum(np.logical_and(pred == 0, label == 0))
    fp = np.sum(np.logical_and(pred == 1, label == 0))
    fn = np.sum(np.logical_and(pred == 0, label == 1))
    return tp, tn, fp, fn

def meta_test(cfg, backbone_net, feature_shape, criterion, device, meta_test_set, bg_test_set):
    backbone_net.eval()

    # Some assertions and prepare necessary variables
    assert cfg.DATASET.dataset == "mini_imagenet"
    num_shots = cfg.META_TEST.shot
    assert num_shots != -1

    # Following PANet, we use five consistent random seeds.
    # For each seed, 1000 (S, Q) pairs are sampled.

    # Meta Test!
    np.random.seed(1221)
    random.seed(1221)
    acc_list = []
    precision_list = []
    recall_list = []
    total_tp, total_tn, total_fp, total_fn = (0, 0, 0, 0)
    meta_loader = get_fs_clf_fg_bg_loader(meta_test_set, bg_test_set, 600, num_shots, 15)
    for j, meta_test_batch in tqdm(enumerate(meta_loader)):
        meta_test_batch = meta_test_batch[0]

        if 'protonet' in cfg.name:
            tp, tn, fp, fn = meta_test_one_proto(cfg, backbone_net, criterion, feature_shape, device, meta_test_batch)
        else:
            tp, tn, fp, fn = meta_test_one(cfg, backbone_net, criterion, feature_shape, device, meta_test_batch)

        acc = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp + 1e-5)
        recall = tp / (tp + fn + 1e-5)

        precision_list.append(precision)
        recall_list.append(recall)

        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn
        
        # Gather episode-wise statistics
        acc_list.append(acc)

    # Gather test numbers
    print("Overall Acc Mean: {:.4f} Std: {:.4f}".format(np.mean(acc_list), np.std(acc_list)))
    print("Overall Precision Mean: {:.4f} Std: {:.4f}".format(np.mean(precision_list), np.std(precision_list)))
    print("Overall Recall Mean: {:.4f} Std: {:.4f}".format(np.mean(recall_list), np.std(recall_list)))
    print("Total TP: {} TN: {} FP: {} FN: {}".format(total_tp, total_tn, total_fp, total_fn))

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
    _, meta_test_set = dataset.meta_dispatcher(cfg)
    bg_test_set = get_bg_test_set(cfg)

    print("Meta test set contains {} data points.".format(len(meta_test_set)))

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
    
    print("Initializing backbone with pretrained weights from: {}".format(args.load))
    trained_weight_dict = torch.load(args.load, map_location=device_str)
    if cfg.BACKBONE.network == "panet_vgg16" and False:
        print("SPECIAL TREATMENT FOR PANET")
        new_dict = {}
        for k in trained_weight_dict:
            new_dict[k[24:]] = trained_weight_dict[k]
        backbone_net.load_state_dict(new_dict, strict=True)
    else:
        backbone_net.load_state_dict(trained_weight_dict['backbone'], strict=True)

    criterion = loss.dispatcher(cfg)

    meta_test(cfg, backbone_net, feature_shape, criterion, device, meta_test_set, bg_test_set)

if __name__ == '__main__':
    main()
