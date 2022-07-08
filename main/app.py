from glob import glob
import __init_lib_path
import errno
from logging import exception
from flask import jsonify
from flask import Flask, render_template, Response, request
import json
import numpy as np
import cv2
import io
from PIL import Image
import pickle
import torch
from config_guard import cfg, update_config_from_yaml
import dataset
import backbone
import classifier
import loss
import trainer
import utils
import argparse
from dataset.coco import get_train_set, get_val_set
import torchvision as tv
from flask_cors import CORS
import base64
from torch.nn.functional import interpolate
import os

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360

#Initialize the Flask app
app = Flask(__name__)
# CORS(app)
camera = cv2.VideoCapture(0)
camera.set(3, IMAGE_WIDTH)
camera.set(4, IMAGE_HEIGHT)
my_trainer = None
normalizer = None

def parse_args():
    parser = argparse.ArgumentParser(description = "Roger's Deep Learning Playground")
    parser.add_argument('--cfg', help = "specify particular yaml configuration to use", required = True,
        default = "configs/mnist_torch_official.yaml", type = str)
    parser.add_argument('--load', help="specify saved checkpoint to evaluate", required=True, type=str)
    parser.add_argument('--jit_trace', help='Trace and serialize trained network. Overwrite other options', action='store_true')
    parser.add_argument('--webcam', help='real-time evaluate using default webcam', action='store_true')
    parser.add_argument('--visfreq', help="visualize results for every n examples in test set",
        required=False, default=99999999999, type=int)
    parser.add_argument("--opts", help="Command line options to overwrite configs", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    return args

def gen_frames_with_pred():  
    global my_trainer
    while True:
        my_trainer.backbone_net.eval()
        my_trainer.post_processor.eval()
        success, frame = camera.read()
        
        if not success:
            break
        else:
            #print("inferencing")
            rgb_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_chw = torch.tensor(rgb_np).float().permute((2, 0, 1))
            img_chw = img_chw / 255 # norm to 0-1
            img_chw = normalizer(img_chw)
            # img_bchw = img_chw.view((1,) + img_chw.shape)
            # pred_map = my_trainer.infer_one(img_bchw).cpu().numpy()[0]
            pred_map = my_trainer.infer_one_aot(rgb_np)
            #print(pred_map.max(), len(my_trainer.class_names))
            label_vis = utils.visualize_segmentation(my_trainer.cfg, img_chw, pred_map, my_trainer.class_names)
            frame = cv2.cvtColor(label_vis, cv2.COLOR_RGB2BGR)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 

def gen_frames():  
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            #print("inferencing")
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 


# APIs:
# 1. camera view video stream (get/yield)
#       Video.js format
# 2. inference view video stream (contains segmentations)(get/yield)
# 3. Trigger fine tune
    '''
    request sample:
    files:
    {
        "file": "SDJFOIDS87x..." (pickle),
        "mask": "[[True False True True .....]]" (pickle) 
        "label": "car" (bytes)
    }
    '''

@app.route('/pred_cam', methods=['GET'])
def get_pred_cam_stream():
    return Response(gen_frames_with_pred(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera', methods=['GET'])
def get_camera_stream():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/snapshot', methods=['GET'])
def take_snapshot():
    if camera.isOpened():
        success, frame = camera.read()
        if not success:
            response = jsonify({'status': "failure"})
            response.headers.add('Access-Control-Allow-Origin', '*')
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            # frame_arr = np.array(buffer)
            # im = Image.fromarray(frame_arr)
            # im_byte = io.BytesIO()
            # im.save(im_byte, format='JPEG')
            frame = buffer.tobytes()
            my_string = str(base64.b64encode(frame))
            response = jsonify({"img": my_string[2:-1]})
            response.headers.add('Access-Control-Allow-Origin', '*')
        return response

@app.route('/trigger_finetune', methods=['POST'])
def trigger_fine_tune():
    img_bytes = request.files['file'].read()
    mask_bytes = request.files['mask'].read()
    label = request.files['label'].read()
    label = label.decode("utf-8") 

    img = pickle.loads(img_bytes)
    mask = pickle.loads(mask_bytes)
    ref_img = np.array(img, dtype='uint8')
    img = torch.LongTensor(img)/255.0
    mask = torch.LongTensor(mask)
    # print("image shape: ", img.shape, "mask shape: ", mask.shape)
    my_trainer.my_aot_segmenter.reset_engine()
    print("reset VOS engine")
    my_trainer.my_aot_segmenter.add_reference_frame(ref_img, mask.cpu().numpy())
    print("added reference frame, start inference with VOS result")
    # img = interpolate(img, scale_factor=0.5)
    # mask = interpolate(mask, scale_factor=0.5)
    img = torch.permute(img, (2, 0, 1))
    print("start adapt a single shot with label: ", label)
    print("image shape: ", img.shape, "mask shape: ", mask.shape)
    # print(img[0, 0, 0], mask[0, 0])

    my_trainer.novel_adapt_single(img, mask, label, blocking=True)

    print("switch back to segmentation model inference only")
    my_trainer.my_aot_segmenter.frame_cnt = 0 # switch back to segmentation model inference only
    response = jsonify({'status': "success"})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


def main():
    #set up our continual learning model
    global my_trainer
    global normalizer

    args = parse_args()
    update_config_from_yaml(cfg, args)

    if cfg.name != "GIFS_scannet25k_from_coco":
        print("wrong cfg for backend server!")
        return

    device = utils.guess_device()
    print("We are now using: ", device)

    torch.manual_seed(cfg.seed)
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
    os.chdir("/home/eason/code_base/dl_codebase/")
    print("Initializing backbone with trained weights from: {}".format(args.load))
    my_trainer.load_model(args.load)

    normalizer = tv.transforms.Normalize(mean=cfg.DATASET.TRANSFORM.TRAIN.TRANSFORMS_DETAILS.NORMALIZE.mean,
                std=cfg.DATASET.TRANSFORM.TRAIN.TRANSFORMS_DETAILS.NORMALIZE.sd)
                
    #start our flask backend
    app.run(debug=False, port=7000)

main()