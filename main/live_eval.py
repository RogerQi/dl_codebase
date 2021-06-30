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
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from IPython import embed

def parse_args():
    parser = argparse.ArgumentParser(description = "Roger's Deep Learning Playground")
    parser.add_argument('--cfg', help="specify particular yaml configuration to use", required=True,
        default="configs/mnist_torch_official.taml", type = str)
    parser.add_argument('--load', help="specify saved checkpoint to evaluate", required=True, type=str)
    args = parser.parse_args()

    return args

def test(cfg, model, post_processor, device, class_names_list):
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
    
    img_idx = 1

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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                output = post_processor(feature, ori_spatial_res)
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
        elif key_press == ord('s'):
            # Save image for segmentation!
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite("new_object_{}.jpg".format(img_idx), frame)
            img_idx += 1
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


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

    class_names_list = test_set.dataset.CLASS_NAMES_LIST
    test(cfg, backbone_net, post_processor, device, class_names_list)


if __name__ == '__main__':
    main()