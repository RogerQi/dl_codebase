import __init_lib_path
from config_guard import cfg, update_config_from_yaml
import dataset
import backbone
import classifier
import loss
import utils

import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description = "Roger's Deep Learning Playground")
    parser.add_argument('--cfg', help="specify particular yaml configuration to use", required=True,
        default="configs/mnist_torch_official.taml", type = str)
    parser.add_argument('--load', help="specify saved checkpoint to evaluate", required=True, type=str)
    parser.add_argument('--visfreq', help="visualize results for every n examples in test set",
        required=False, default=99999999999, type=int)
    args = parser.parse_args()

    return args

def test(cfg, model, post_processor, criterion, device, test_loader, visfreq):
    model.eval()
    post_processor.eval()
    test_loss = 0
    correct = 0
    pixel_acc_list = []
    iou_list = []
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            feature = model(data)
            output = post_processor(feature)
            test_loss += criterion(output, target).item()  # sum up batch loss
            if cfg.task == "classification":
                pred = output.argmax(dim = 1, keepdim = True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                # TODO: save classified images with raw image as content and
                # human readable label as filenames
            elif cfg.task == "semantic_segmentation":
                pred_map = output.max(dim = 1)[1]
                batch_acc, _ = utils.compute_pixel_acc(pred_map, target, fg_only=cfg.METRIC.SEGMENTATION.fg_only)
                pixel_acc_list.append(float(batch_acc))
                for i in range(pred_map.shape[0]):
                    pred_np = np.array(pred_map[i].cpu())
                    target_np = np.array(target[i].cpu(), dtype=np.int64)
                    iou = utils.compute_iou(pred_np, target_np, cfg.num_classes, fg_only=cfg.METRIC.SEGMENTATION.fg_only)
                    iou_list.append(float(iou))
                    if (i + 1) % visfreq == 0:
                        cv2.imwrite("{}_{}_pred.png".format(idx, i), pred_np)
                        cv2.imwrite("{}_{}_label.png".format(idx, i), target_np)
                        # Visualize RGB image as well
                        ori_rgb_np = np.array(data[i].permute((1, 2, 0)).cpu())
                        if 'normalize' in cfg.DATASET.TRANSFORM.TEST.transforms:
                            rgb_mean = cfg.DATASET.TRANSFORM.TEST.TRANSFORMS_DETAILS.NORMALIZE.mean
                            rgb_sd = cfg.DATASET.TRANSFORM.TEST.TRANSFORMS_DETAILS.NORMALIZE.sd
                            ori_rgb_np = (ori_rgb_np * rgb_sd) + rgb_mean
                        assert ori_rgb_np.max() <= 1.1, "Max is {}".format(ori_rgb_np.max())
                        ori_rgb_np[ori_rgb_np >= 1] = 1
                        ori_rgb_np = (ori_rgb_np * 255).astype(np.uint8)
                        # Convert to OpenCV BGR
                        ori_rgb_np = cv2.cvtColor(ori_rgb_np, cv2.COLOR_RGB2BGR)
                        cv2.imwrite("{}_{}_ori.jpg".format(idx, i), ori_rgb_np)
            else:
                raise NotImplementedError

    test_loss /= len(test_loader.dataset)

    if cfg.task == "classification":
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    elif cfg.task == "semantic_segmentation":
        print('\nTest set: Average loss: {:.4f}, Mean Pixel Accuracy: {:.4f}, Mean IoU {:.4f}\n'.format(
            test_loss, np.mean(pixel_acc_list), np.mean(iou_list)))
    else:
        raise NotImplementedError


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

    print("Test/Val set contains {} data points.".format(len(test_set)))

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.TEST.batch_size, **kwargs)

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

    test(cfg, backbone_net, post_processor, criterion, device, test_loader, args.visfreq)


if __name__ == '__main__':
    main()