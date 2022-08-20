import numpy as np
import json
from tqdm import tqdm
import os
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt
from panopticapi.utils import rgb2id

# path to the folder containing raw images
pic_folder = '/data/COCO2017/train2017/'
# path to the json file containing panoptic annotation
json_file = '/data/COCO2017/annotations/panoptic_train2017.json'
# Path to the folder containing panoptic segmentation images
seg_folder = '/data/COCO2017/annotations/panoptic_train2017/'
# Folder where all semantic images will be saved to
semantic_seg_folder = '/data/COCO2017/annotations/panoptic_semantic_train2017/'

def main():
    with open(json_file, 'r') as f:
        coco_p = json.load(f)

    for i in tqdm(range(len(coco_p['annotations']))):
        annotation = coco_p['annotations'][i]
        img_name = annotation['file_name']
        seg_img_path = os.path.join(seg_folder, img_name)
        # Load seg image
        seg_img = np.array(Image.open(seg_img_path), dtype=np.uint8)
        # decode to id that can be looked up in panoptic segmentation
        id_mask = rgb2id(seg_img)
        # semantic mask map
        semantic = np.zeros(id_mask.shape, dtype=np.uint8)
        for segm_info in annotation['segments_info']:
            cat_id = segm_info['category_id']
            mask = (id_mask == segm_info['id'])
            semantic[mask] = cat_id
        semantic_path = os.path.join(semantic_seg_folder, img_name)
        Image.fromarray(semantic).save(semantic_path)

if __name__ == '__main__':
    main()