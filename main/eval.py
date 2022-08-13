from operator import gt
from torch import logical_or
import __init_lib_path
from config_guard import cfg, update_config_from_yaml
import dataset
import backbone
import classifier
from vision_hub.interactive_seg.ritm import ritm_segmenter
import loss
import trainer
import utils
import vision_hub
from dataset.scannet_seq_reader import scannet_scene_reader
import cv2

import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision as tv

import os
import json
import zipfile
import pickle
from tqdm import trange, tqdm
from scipy.stats import hmean

EVAL_SCENE_NAME = ['scene0050_00', 'scene0565_00', 'scene0462_00', 'scene0144_00', 'scene0593_00']
SCANNET_PATH = "/home/randomgraph/yichen14/dl_codebase/data/scannet_v2"

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

def gen_annotation_interval(model, obj_scene_map):
    """
        generate annotation intervals that satisfy our criterions for each scene
    """
    num_clicks_spent = {}
    interest_obj_list = sorted(list(obj_scene_map.keys()))

    #some constants
    minimum_instance_size = 2000

    video_out_path = '/home/randomgraph/yichen14/dl_codebase/visualization/videos'

    for canonical_obj_name in interest_obj_list:
        
        adaptation_scene_list = obj_scene_map[canonical_obj_name][:-2]

        for scene_name in adaptation_scene_list:
            print("generate annotation interval for scene: {} object: {}".format(scene_name, canonical_obj_name))
            # visulization:
            out = cv2.VideoWriter(os.path.join(video_out_path,'annotation_interval_2000', 'annotation_interval_{}_{}.mp4'.format(canonical_obj_name, scene_name)),cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 480))

            num_clicks_spent[scene_name] = 0
            my_seq_reader = scannet_scene_reader(SCANNET_PATH, scene_name)
            inst_name_map = my_seq_reader.get_inst_name_map()
            first_seen_dict = {} # record first seen frame of each instance
            annotated_frame_per_inst = {}

            for i in trange(len(my_seq_reader)):
                data_dict = my_seq_reader[i]
                img = data_dict['color']
                mask = data_dict['semantic_label']
                inst_map = data_dict['inst_label']
                img_chw = torch.tensor(img).float().permute((2, 0, 1))
                img_chw = img_chw / 255 # norm to 0-1
                img_chw = model.normalizer(img_chw)

                could_annotate = False
                if mask.max() >= model.cfg.num_classes:

                    unique_inst_list = np.unique(inst_map)
                    for inst in unique_inst_list:
                        if inst == 0: continue # background
                        if inst_name_map[inst] == canonical_obj_name:

                            if inst not in first_seen_dict:
                                first_seen_dict[inst] = i
                            else:
                                # check criterion
                                if i < first_seen_dict[inst] + 30:
                                    continue
                                
                                pixel_cnt = np.sum(inst_map == inst)
                                if pixel_cnt < minimum_instance_size:
                                    continue

                                no_boundary_cnt = np.sum(inst_map[1:-1,1:-1] == inst)
                                if no_boundary_cnt != pixel_cnt:
                                    # pixel locates at boundary
                                    continue


                                gt_mask = np.zeros_like(mask)
                                for inst in unique_inst_list:
                                    if inst == 0: continue
                                    if inst_name_map[inst] == canonical_obj_name:
                                        instance_mask = (inst_map == inst).astype(np.uint8)
                                        gt_mask = np.logical_or(gt_mask, instance_mask).astype(np.uint8)

                                label_vis = utils.visualize_segmentation(model.cfg, img_chw, gt_mask, model.class_names)
                                frame = cv2.cvtColor(label_vis, cv2.COLOR_RGB2BGR)
                                frame = cv2.putText(frame, str(i), (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                                out.write(frame)
                                could_annotate = True
                                if inst not in annotated_frame_per_inst:
                                    annotated_frame_per_inst[inst] = [i]
                                else:
                                    annotated_frame_per_inst[inst].append(i)
                if not could_annotate:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    img = cv2.putText(img, str(i), (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    out.write(img)
            out.release()
            for process_name in model.process_pool:
                model.process_pool[process_name].join()

def evaluation(model, segmenter, obj_scene_map):
    annotation_frames = {}
    annotation_frames['scene0329_00'] = [1280]
    annotation_frames['scene0565_00'] = [229] # 119 need to be disscussed
    annotation_frames['scene0644_00'] = [402, 968, 1069]
    annotation_frames['scene0207_00'] = [527, 867, 1209]

    num_clicks_spent = {}
    online_inference_latency_dict = {}
    interest_obj_list = sorted(list(obj_scene_map.keys()))
    
    scores = {} # scores for each object

    #some constants
    minimum_instance_size = 5000
    max_clicks = 20
    iou_thresh = 0.85
    delay_violation_cnt = 0

    model.take_snapshot()

    video_out_path = '/home/randomgraph/yichen14/dl_codebase/visualization/videos'

    for canonical_obj_name in ['computer tower']:
        print("------------------------------------------------")
        model.restore_last_snapshot()

        print("now eval on {}, it contains {} scenes, use last two scenes for generalization test".format(canonical_obj_name, len(obj_scene_map[canonical_obj_name])))

        # set last two sequences as generalization test seqs
        adaptation_scene_list = obj_scene_map[canonical_obj_name][:-2]
        generalization_scene_list = obj_scene_map[canonical_obj_name][-2:]

        scores[canonical_obj_name] = {}

        # adaptation score
        print("------------------------------------------------")
        print("Calculating short-term adaptation score...")
        model.prepare_for_eval()
        c = 0
        adapt_ious = []
        adapt_latency = []
        adapt_clicks = []

        for scene_name in adaptation_scene_list:

            # visulization:
            out_inf = cv2.VideoWriter(os.path.join(video_out_path, 'adaptation_inf_{}_{}.mp4'.format(canonical_obj_name, scene_name)),cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 480))
            out_gt = cv2.VideoWriter(os.path.join(video_out_path, 'adaptation_gt_{}_{}.mp4'.format(canonical_obj_name, scene_name)),cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 480))

            # model.prepare_for_eval()
            c+=1
            print("Evaluate on {}, {}/{} of adaptation seq".format(scene_name, c, len(adaptation_scene_list)))
            print(model.class_names)
            scores[canonical_obj_name]["adaptation"] = {}
            scores[canonical_obj_name]["generalization"] = {}

            num_clicks_spent[scene_name] = 0
            my_seq_reader = scannet_scene_reader(SCANNET_PATH, scene_name)
            inst_name_map = my_seq_reader.get_inst_name_map()
            online_inference_latency_dict[scene_name] = []
            first_seen_dict = {} # record first seen frame of each instance
            annotated_frame_per_inst = {}
            
            tp_cnt, fp_cnt, tn_cnt, fn_cnt = 0, 0, 0, 0

            for i in trange(len(my_seq_reader)):
                data_dict = my_seq_reader[i]
                img = data_dict['color']
                mask = data_dict['semantic_label']
                inst_map = data_dict['inst_label']

                start_cp = time.time()
                combined_pred_map, vos_pred_map = model.infer_one_aot(img)
                latency = time.time() - start_cp
                online_inference_latency_dict[scene_name].append(latency)
                if latency < 0.0333:
                    latency = 0.0333
                
                # if system halt IoU = 0
                if latency > 10:
                    continue

                if canonical_obj_name in model.class_names:
                    unique_inst_list = np.unique(inst_map)
                    gt_mask = np.zeros_like(vos_pred_map)
                    for inst in unique_inst_list:
                        if inst == 0: continue
                        if inst_name_map[inst] == canonical_obj_name:
                            instance_mask = (inst_map == inst).astype(np.uint8)
                            gt_mask = np.logical_or(gt_mask, instance_mask).astype(np.uint8)

                    interest_cls_idx = model.class_names.index(canonical_obj_name)
                    gt_mask[gt_mask>0] = interest_cls_idx

                    # Seen this object before, eval
                    metrics_dict = utils.compute_binary_metrics(vos_pred_map, gt_mask, class_idx=interest_cls_idx)
                    tp_cnt += metrics_dict['tp']# * 0.0333/latency
                    fp_cnt += metrics_dict['fp']
                    tn_cnt += metrics_dict['tn']
                    fn_cnt += metrics_dict['fn']

                    # visualize frame for debug purpose
                    img_chw = torch.tensor(img).float().permute((2, 0, 1))
                    img_chw = img_chw / 255 # norm to 0-1
                    img_chw = model.normalizer(img_chw)
                    label_vis = utils.visualize_segmentation(model.cfg, img_chw, vos_pred_map, model.class_names)
                    frame = cv2.cvtColor(label_vis, cv2.COLOR_RGB2BGR)
                    frame = cv2.putText(frame, str(i), (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.imwrite("/home/randomgraph/yichen14/dl_codebase/visualization/adaptation_inference.jpg", frame)
                    out_inf.write(frame)

                    label_vis = utils.visualize_segmentation(model.cfg, img_chw, gt_mask, model.class_names)
                    frame = cv2.cvtColor(label_vis, cv2.COLOR_RGB2BGR)
                    frame = cv2.putText(frame, str(i), (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.imwrite("/home/randomgraph/yichen14/dl_codebase/visualization/adaptation_gt.jpg", frame)
                    out_gt.write(frame)

                if mask.max() >= model.cfg.num_classes:

                    unique_inst_list = np.unique(inst_map)
                    for inst in unique_inst_list:
                        if inst == 0: continue # background
                        if inst_name_map[inst] == canonical_obj_name:

                            if inst not in first_seen_dict:
                                first_seen_dict[inst] = i
                            else:
                                # check criterion
                                if i < first_seen_dict[inst] + 30:
                                    continue

                                if inst in annotated_frame_per_inst and len(annotated_frame_per_inst[inst]) >= 5:
                                    continue

                                if inst in annotated_frame_per_inst and i < annotated_frame_per_inst[inst][-1] + 300:
                                    continue

                                if canonical_obj_name in model.class_names:
                                    if metrics_dict['iou'] > 0.7:
                                        continue
                                
                                pixel_cnt = np.sum(inst_map == inst)
                                if pixel_cnt < minimum_instance_size:
                                    continue

                                no_boundary_cnt = np.sum(inst_map[1:-1,1:-1] == inst)
                                if no_boundary_cnt != pixel_cnt:
                                    # pixel locates at boundary
                                    continue

                                # All criterion passed; now we can provide annotation
                                # select relevant instance
                                if i in annotation_frames[scene_name]:
                                    print("Found the novel object that satisfy all criterions, object instance index: {}".format(inst))
                                    instance_mask = (inst_map == inst).astype(np.uint8)
                                    provided_mask, num_click = segmenter.auto_eval(img, instance_mask, max_clicks=max_clicks, iou_thresh=iou_thresh)
                                    print("Spent {} clicks".format(num_click))
                                    num_clicks_spent[scene_name] += num_click

                                    provided_mask = torch.tensor(provided_mask).int()
                                    model.novel_adapt_single_w_aot(img, provided_mask, canonical_obj_name, blocking=False)

                                    # record annotated frame for this instance
                                    if inst not in annotated_frame_per_inst:
                                        annotated_frame_per_inst[inst] = [i]
                                    else:
                                        annotated_frame_per_inst[inst].append(i)
                                    
                                    img_chw = torch.tensor(img).float().permute((2, 0, 1))
                                    img_chw = img_chw / 255 # norm to 0-1
                                    img_chw = model.normalizer(img_chw)
                                    label_vis = utils.visualize_segmentation(model.cfg, img_chw, provided_mask, model.class_names)
                                    frame = cv2.cvtColor(label_vis, cv2.COLOR_RGB2BGR)
                                    cv2.imwrite("/home/randomgraph/yichen14/dl_codebase/visualization/annotations/annotation_frame_{}_{}_{}.jpg".format(scene_name, i, minimum_instance_size), frame)

            out_gt.release()
            out_inf.release()
            if num_clicks_spent[scene_name] == 0:
                print("No annotation provided for {}".format(scene_name))
            print("============================")
            print(scene_name)
            print("| Object IoU: {:.4f}".format(tp_cnt / (tp_cnt + fp_cnt + fn_cnt + 1e-10)))
            print("| Avg latency: {:.4f} w/ std: {:.4f}".format(
                np.mean(online_inference_latency_dict[scene_name]), np.std(online_inference_latency_dict[scene_name])))
            print("============================")
            # print("Main inference completed. Waiting for processes {} to finish".format(model.process_pool.keys()))

            adapt_ious.append(tp_cnt / (tp_cnt + fp_cnt + fn_cnt + 1e-10))
            adapt_latency.append(np.mean(online_inference_latency_dict[scene_name]))
            adapt_clicks.append(num_clicks_spent[scene_name])

            for process_name in model.process_pool:
                model.process_pool[process_name].join()
            
        print(adapt_ious)
        scores[canonical_obj_name]["adaptation"]["IoU"] = hmean(adapt_ious)
        scores[canonical_obj_name]["adaptation"]["latency"] = np.mean(adapt_latency)
        scores[canonical_obj_name]["adaptation"]["clicks"] = np.mean(adapt_clicks)

        print(scores[canonical_obj_name]["adaptation"])

        print("------------------------------------------------")
        print("Calculating generalization score...")
        model.prepare_for_eval()
        general_ious = []
        general_latency = []
        for scene_name in generalization_scene_list:

            # visulization:
            out_inf = cv2.VideoWriter(os.path.join(video_out_path, 'general_inf_{}_{}.mp4'.format(canonical_obj_name, scene_name)),cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 480))
            out_gt = cv2.VideoWriter(os.path.join(video_out_path, 'general_gt_{}_{}.mp4'.format(canonical_obj_name, scene_name)),cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 480))

            # model.prepare_for_eval()
            print("Evaluate on {}".format(scene_name))
            # num_clicks_spent[scene_name] = 0
            my_seq_reader = scannet_scene_reader(SCANNET_PATH, scene_name)
            inst_name_map = my_seq_reader.get_inst_name_map()
            online_inference_latency_dict[scene_name] = []
            annotated_frame_per_inst = {}

            tp_cnt, fp_cnt, tn_cnt, fn_cnt = 0, 0, 0, 0

            for i in trange(len(my_seq_reader)):
                data_dict = my_seq_reader[i]
                img = data_dict['color']
                mask = data_dict['semantic_label']
                inst_map = data_dict['inst_label']

                start_cp = time.time()
                combined_pred_map, _ = model.infer_one_aot(img)
                latency = time.time() - start_cp
                online_inference_latency_dict[scene_name].append(time.time() - start_cp)
                if latency < 0.0333:
                    latency = 0.0333

                unique_inst_list = np.unique(inst_map)
                gt_mask = np.zeros_like(combined_pred_map)
                for inst in unique_inst_list:
                    if inst == 0: continue
                    if inst_name_map[inst] == canonical_obj_name:
                        instance_mask = (inst_map == inst).astype(np.uint8)
                        gt_mask = np.logical_or(gt_mask, instance_mask).astype(np.uint8)

                interest_cls_idx = model.class_names.index(canonical_obj_name)
                gt_mask[gt_mask>0] = interest_cls_idx

                # Seen this object before, eval
                metrics_dict = utils.compute_binary_metrics(combined_pred_map, gt_mask, class_idx=interest_cls_idx)
                tp_cnt += metrics_dict['tp']# * 0.0333/latency
                fp_cnt += metrics_dict['fp']
                tn_cnt += metrics_dict['tn']
                fn_cnt += metrics_dict['fn']

                # visualize frame for debug purpose
                img_chw = torch.tensor(img).float().permute((2, 0, 1))
                img_chw = img_chw / 255 # norm to 0-1
                img_chw = model.normalizer(img_chw)
                label_vis = utils.visualize_segmentation(model.cfg, img_chw, combined_pred_map, model.class_names)
                frame = cv2.cvtColor(label_vis, cv2.COLOR_RGB2BGR)
                cv2.imwrite("/home/randomgraph/yichen14/dl_codebase/visualization/general_inference.jpg", frame)
                out_inf.write(frame)
                
                label_vis = utils.visualize_segmentation(model.cfg, img_chw, gt_mask, model.class_names)
                frame = cv2.cvtColor(label_vis, cv2.COLOR_RGB2BGR)
                cv2.imwrite("/home/randomgraph/yichen14/dl_codebase/visualization/general_gt.jpg", frame)
                out_gt.write(frame)

            out_gt.release()
            out_inf.release()
            print("============================")
            print(scene_name)
            print(tp_cnt)
            print("| Object IoU: {:.4f}".format(tp_cnt / (tp_cnt + fp_cnt + fn_cnt + 1e-10)))
            print("| Avg latency: {:.4f} w/ std: {:.4f}".format(
                    np.mean(online_inference_latency_dict[scene_name]), np.std(online_inference_latency_dict[scene_name])))
            print("============================")
            general_ious.append(tp_cnt / (tp_cnt + fp_cnt + fn_cnt + 1e-10))
            general_latency.append(np.mean(online_inference_latency_dict[scene_name]))
        
        scores[canonical_obj_name]["generalization"]["IoU"] = hmean(general_ious)
        scores[canonical_obj_name]["generalization"]["latency"] = np.mean(general_latency)

        print(scores[canonical_obj_name]["generalization"])

        scores[canonical_obj_name]["IoU Score"] = hmean([scores[canonical_obj_name]["generalization"]["IoU"], scores[canonical_obj_name]["adaptation"]["IoU"]])

        print("Total clicks expanded: {}".format(num_clicks_spent))
        print(f"Total delay violation: {delay_violation_cnt}")
        print("eval on old dataset to test catastrophic forgetting")
        class_iou, _ = model.eval_on_loader(model.val_loader, 22)
        print("Base IoU after adaptation")
        print(np.mean(class_iou[:-1]))

        print(scores[canonical_obj_name]["IoU Score"])

    print(scores)


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
    # os.chdir("/home/randomgraph/yichen14/dl_codebase")
    args = parse_args()
    update_config_from_yaml(cfg, args)
    device = utils.guess_device()
    torch.manual_seed(cfg.seed)
    
    # set up segmenter
    my_ritm_segmenter = vision_hub.interactive_seg.ritm_segmenter()

    my_vos_engine = vision_hub.aotb.aot_segmenter()

    # set up model (for now is the trainer)
    my_trainer = get_trainer(cfg, device, args.load)
    
    with open('/home/randomgraph/yichen14/dl_codebase/metadata/scannet_map.pkl', 'rb') as f:
        obj_scene_map = pickle.load(f)
        evaluation(model=my_trainer, segmenter=my_ritm_segmenter, obj_scene_map=obj_scene_map)
        # gen_annotation_interval(model=my_trainer, obj_scene_map=obj_scene_map)

if __name__ == '__main__':
    main()
