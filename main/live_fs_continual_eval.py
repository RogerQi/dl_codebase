import __init_lib_path
from config_guard import cfg, update_config_from_yaml
import dataset
import backbone
import classifier
import loss
import utils
import os
import argparse
import numpy as np
import cv2
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image

from IPython import embed

to_tensor_func = torchvision.transforms.ToTensor()

def parse_args():
    parser = argparse.ArgumentParser(description = "Roger's Deep Learning Playground")
    parser.add_argument('--cfg', help="specify particular yaml configuration to use", required=True,
        default="configs/mnist_torch_official.taml", type = str)
    parser.add_argument('--load', help="specify saved checkpoint to evaluate", required=True, type=str)
    args = parser.parse_args()

    return args

def test(cfg, model, post_processor, criterion, device, class_names_list, novel_cutoff):
    # Put model to eval state
    model.eval()
    post_processor.eval()

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Prep data pre-processor
    image_to_tensor = torchvision.transforms.ToTensor()
    normalizer = torchvision.transforms.Normalize(cfg.DATASET.TRANSFORM.TEST.TRANSFORMS_DETAILS.NORMALIZE.mean,
                                    cfg.DATASET.TRANSFORM.TEST.TRANSFORMS_DETAILS.NORMALIZE.sd)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        assert frame.shape == (480, 640, 3)
        H, W, C = frame.shape
        assert W >= H
        strip_size = (W - H) // 2
        frame = frame[:, strip_size:-strip_size, :]
        H, W, C = frame.shape
        assert H == W
        frame = cv2.resize(frame, (480, 480), interpolation = cv2.INTER_LINEAR)
        # Model Inference
        data = image_to_tensor(frame) # 3 x H x W
        assert data.shape == (C, H, W)
        assert data.min() >= 0
        assert data.max() <= 1
        data = normalizer(data)
        data = data.view((1,) + data.shape) # B x 3 x H x W
        with torch.no_grad():
            # Forward Pass
            data = data.to(device)
            feature = model(data)
            if cfg.task == "classification":
                output = post_processor(feature)
            elif cfg.task == "semantic_segmentation" or cfg.task == "few_shot_semantic_segmentation_fine_tuning":
                ori_spatial_res = data.shape[-2:]
                output = post_processor(feature, ori_spatial_res) # B x num_classes x H x W
                # manual cutoff
                output[:, -1, :, :][output[:, -1, :, :] < novel_cutoff] = 0
            else:
                raise NotImplementedError
            # Visualization
            if cfg.task == "classification":
                pred = output.argmax(dim = 1, keepdim = True)
                raise NotImplementedError
            elif cfg.task == "semantic_segmentation" or cfg.task == "few_shot_semantic_segmentation_fine_tuning":
                pred_map = output.max(dim = 1)[1]
                assert pred_map.shape[0] == 1
                pred_np = np.array(pred_map[0].cpu())
                predicted_label = utils.visualize_segmentation(cfg, data[0], pred_np, class_names_list)
            else:
                raise NotImplementedError
        # Display the resulting frame
        cv2.imshow('raw_image', frame)
        cv2.imshow('predicted_label', predicted_label)
        key_press = cv2.waitKey(1)
        if key_press == ord('q'):
            # Quit!
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def masked_average_pooling(mask_b1hw, feature_bchw, normalization):
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
    _, test_set = dataset.dispatcher(cfg)

    normalizer = torchvision.transforms.Normalize(cfg.DATASET.TRANSFORM.TEST.TRANSFORMS_DETAILS.NORMALIZE.mean,
                                    cfg.DATASET.TRANSFORM.TEST.TRANSFORMS_DETAILS.NORMALIZE.sd)

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
    post_processor = classifier.dispatcher(cfg, feature_shape)
    
    post_processor = post_processor.to(device)
    
    print("Initializing backbone with pretrained weights from: {}".format(args.load))
    trained_weight_dict = torch.load(args.load, map_location=device_str)
    backbone_net.load_state_dict(trained_weight_dict["backbone"], strict=True)
    post_processor.load_state_dict(trained_weight_dict["head"], strict=True)

    criterion = loss.dispatcher(cfg)

    class_names_list = test_set.dataset.CLASS_NAMES_LIST
    class_names_list.append("new_object")

    # Train with new object
    file_list = os.listdir('new_objects')
    num_shots = len(file_list) // 2

    img_list = []
    target_list = []
    for i in range(1, num_shots + 1):
        img_path = os.path.join('new_objects', '{}.jpg'.format(i))
        mask_path = os.path.join('new_objects', '{}.png'.format(i))
        img = np.array(Image.open(img_path).convert('RGB'))
        img = to_tensor_func(img)
        img = normalizer(img)
        mask = np.array(Image.open(mask_path), dtype=np.uint8) # 0 is unannotated / 1 is foreground
        mask = torch.tensor(mask)
        img_list.append(img)
        target_list.append(mask)

    img_tensor_bchw = torch.stack(img_list)
    target_tensor_bhw = torch.stack(target_list) # 0, 1

    backbone_net.eval()
    with torch.no_grad():
        img_tensor_bchw = img_tensor_bchw.to(device)
        target_tensor_bhw = target_tensor_bhw.to(device)
        assert 1 in target_tensor_bhw
        feature = backbone_net(img_tensor_bchw)
        class_weight_vec = masked_average_pooling(target_tensor_bhw == 1, feature, True).reshape((1, -1, 1, 1))
        ori_post_processor_weight = post_processor.pixel_classifier.class_mat.weight.data
        ori_num_classes = ori_post_processor_weight.shape[0]
        post_processor = classifier.dispatcher(cfg, feature_shape, num_classes=ori_num_classes + 1)
        post_processor = post_processor.to(device)
        post_processor.pixel_classifier.class_mat.weight.data = torch.cat([ori_post_processor_weight, class_weight_vec])
        post_processor.eval()

        # Cutoff grid search
        cutoff_candidates = np.linspace(0, 3, 31)
        best_iou = 0
        best_idx = -1
        for i in range(len(cutoff_candidates)):
            cur_cutoff = cutoff_candidates[i]
            ori_spatial_res = img_tensor_bchw.shape[-2:]
            output = post_processor(feature, ori_spatial_res) # B x num_classes x H x W
            # manual cutoff
            output[:, -1, :, :][output[:, -1, :, :] < cur_cutoff] = 0
            pred_map = output.max(dim = 1)[1]
            iou_list = []
            for j in range(pred_map.shape[0]):
                pred_np = np.array(pred_map[j].cpu())
                target_np = np.array(target_tensor_bhw[j].cpu(), dtype=np.int64)                
                predicted_fg = (pred_np == ori_num_classes + 1 - 1)
                predicted_bg = (pred_np != ori_num_classes + 1 - 1)
                gt_fg = (target_np == 1)
                gt_bg = (target_np == 0)

                tp_cnt = np.logical_and(predicted_fg, gt_fg).sum()
                fp_cnt = np.logical_and(predicted_fg, gt_bg).sum()
                fn_cnt = np.logical_and(predicted_bg, gt_fg).sum()
                tn_cnt = np.logical_and(predicted_bg, gt_bg).sum()

                iou = tp_cnt / (tp_cnt + fp_cnt + fn_cnt)
                iou_list.append(iou)
            
            avg_iou = np.mean(iou_list)
            if avg_iou > best_iou:
                best_iou = avg_iou
                best_idx = i

    assert best_idx != -1
    print("Selected: {}".format(cutoff_candidates[best_idx]))
    test(cfg, backbone_net, post_processor, criterion, device, class_names_list, cutoff_candidates[best_idx] - 1)


if __name__ == '__main__':
    main()