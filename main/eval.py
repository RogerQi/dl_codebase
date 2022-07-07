import __init_lib_path
from config_guard import cfg, update_config_from_yaml
import dataset
import backbone
import classifier
import loss
import trainer
import utils
import vision_hub
from dataset.scannet_seq_reader import scannet_scene_reader

import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision as tv

EVAL_SCENE_NAME = ['scene0050_00', 'scene0565_00', 'scene0462_00', 'scene0144_00', 'scene0593_00']
SCANNET_PATH = "/media/eason/My Passport/data/scannet_v2"

def parse_args():
    parser = argparse.ArgumentParser(description = "Continual learning benchmark")
    parser.add_argument('--cfg', help = "specify particular yaml configuration to use", required = True,
        default = "configs/scannet_25k/deeplabv3.yaml", type = str)
    parser.add_argument('--load', help="specify saved checkpoint to evaluate", required=True, type=str)
    # parser.add_argument('--jit_trace', help='Trace and serialize trained network. Overwrite other options', action='store_true')
    # parser.add_argument('--webcam', help='real-time evaluate using default webcam', action='store_true')
    # parser.add_argument('--visfreq', help="visualize results for every n examples in test set",
    #     required=False, default=99999999999, type=int)
    parser.add_argument("--opts", help="Command line options to overwrite configs", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    return args


def evaluation(model, segmenter, vos_engine=None):
    model.prepare_for_eval()
    normalizer = model.normalizer

    offline_inference_latency_dict = {}
    online_inference_latency_dict = {}
    precision_dict = {}    
    num_clicks_spent = {}
    for scene in EVAL_SCENE_NAME:
        num_clicks_spent[scene] = 0
        seq_reader = scannet_scene_reader(SCANNET_PATH, scene)

        # Blocking offline evaluation to compute reference metrics
        model.take_snapshot()
        offline_inference_latency_dict[scene] = []
        for i in range(len(seq_reader)):
            data_dict = seq_reader[i]
            img = data_dict['color']
            mask = data_dict['semantic_label']
            img_chw = torch.tensor(img).float().permute((2, 0, 1))
            img_chw = img_chw / 255 # norm to 0-1
            img_chw = normalizer(img_chw)
            img_bchw = img_chw.view((1,) + img_chw.shape)

            # computer latency for single inferencing single image
            start_cp = time.time()
            pred_map = model.infer_one(img_bchw).cpu().numpy()[0]
            offline_inference_latency_dict[scene].append(time.time() - start_cp)

            # label_vis = utils.visualize_segmentation(self.cfg, img_chw, pred_map, self.class_names)
            if canonical_obj_name in model.class_names:
                # Seen this object before, eval
                intersection, union = utils.compute_iu(pred_map, mask, num_classes=22, fg_only=True)
                if offline_intersection is None:
                    offline_intersection = intersection
                    offline_union = union
                else:
                    offline_intersection += intersection
                    offline_union += union



    pass

def get_trainer(cfg, device, model_path):
    dataset_module = dataset.dataset_dispatcher(cfg)
    backbone_net = backbone.dispatcher(cfg)
    backbone_net = backbone_net(cfg).to(device)
    feature_shape = backbone_net.get_feature_tensor_shape(device)
    print("Flatten feature length: {}".format(feature_shape))
    post_processor = classifier.dispatcher(cfg, feature_shape)
    
    post_processor = post_processor.to(device)

    criterion = loss.dispatcher(cfg)

    trainer_func = trainer.dispatcher(cfg)
    my_trainer = trainer_func(cfg, backbone_net, post_processor, criterion, dataset_module, device)

    print("Initializing backbone with trained weights from: {}".format(model_path))
    my_trainer.load_model(model_path)

    return my_trainer

def main():
    # --------------------------
    # | Initial set up
    # |  1. Parse arg and yaml
    # |  2. Set device
    # |  3. Set seed
    # --------------------------
    args = parse_args()
    update_config_from_yaml(cfg, args)
    device = utils.guess_device()
    torch.manual_seed(cfg.seed)
    
    # set up segmenter
    my_ritm_segmenter = vision_hub.interactive_seg.ritm_segmenter()

    # set up model (for now is the trainer)
    my_trainer = get_trainer(cfg, device, args.load)
    
    evaluation(model=my_trainer, segmenter=my_ritm_segmenter)

if __name__ == '__main__':
    main()
