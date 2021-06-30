import __init_lib_path
from config_guard import cfg, update_config_from_yaml
import dataset
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
def meta_test(cfg, backbone_net, feature_shape, criterion, device, meta_test_set):
    backbone_net.eval()

    # Some assertions and prepare necessary variables
    assert cfg.task.startswith('few_shot')
    assert cfg.DATASET.dataset == "pascal_5i"
    num_shots = cfg.META_TEST.shot
    assert num_shots != -1

    # Following PANet, we use five consistent random seeds.
    # For each seed, 1000 (S, Q) pairs are sampled.
    np.random.seed(1221)
    seed_list = np.random.randint(0, 99999, size = (5, ))

    meta_test_task = [[None for j in range(1000)] for i in range(5)]
    # Meta Test!
    iou_list = []
    for i in range(len(meta_test_task)):
        np.random.seed(seed_list[i])
        random.seed(seed_list[i])
        for j in tqdm(range(len(meta_test_task[i]))):
            meta_test_batch = meta_test_set.episodic_sample(num_shots)
            meta_test_task[i][j] = meta_test_batch

            if False:
                plt.imshow((meta_test_batch['query_mask_bhw'][0] == 1).cpu().numpy())
                plt.imshow((meta_test_batch['supp_mask_bhw'][0] == 1).cpu().numpy())

            tp_cnt, fp_cnt, fn_cnt, tn_cnt = meta_test_one(cfg, backbone_net, criterion, feature_shape, device, meta_test_batch)
            
            # Gather episode-wise statistics
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

    meta_test(cfg, backbone_net, feature_shape, criterion, device, meta_test_set)

if __name__ == '__main__':
    main()
