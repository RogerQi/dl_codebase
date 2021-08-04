import sys
import os
import json
import numpy as np
import torch
import torchvision
from copy import deepcopy
from torchvision import datasets, transforms
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from .baseset import base_set

from IPython import embed

# 2017 train images normalization constants
#   mean: 0.4700, 0.4468, 0.4076
#   sd: 0.2439, 0.2390, 0.2420

class COCOSeg(datasets.vision.VisionDataset):
    def __init__(self, root, train=True):
        super(COCOSeg, self).__init__(root, None, None, None)
        self.min_area = 0 # 200
        split_name = "train" if train else "val"
        self.annotation_path = os.path.join(root, 'annotations', 'instances_{}2017.json'.format(split_name))
        self.img_dir = os.path.join(root, '{}2017'.format(split_name))
        self.coco = COCO(self.annotation_path)
        self.img_ids = list(self.coco.imgs.keys())

        # COCO class
        class_list = sorted([i for i in self.coco.cats.keys()])

        # The instances labels in COCO dataset is not dense
        # e.g., total 80 classes. Some objects are labeled as 82
        # but they are 73rd class; while none is labeled as 83.
        self.class_map = {}
        for i in range(len(class_list)):
            self.class_map[class_list[i]] = i + 1
        
        # Given a class idx (1-80), self.class_map gives the list of images that contain
        # this class idx
        self.instance_class_map = self.init_class_map()

        self.CLASS_NAMES_LIST = ['background']
        for i in range(len(class_list)):
            cls_name = self.coco.cats[class_list[i]]['name']
            self.CLASS_NAMES_LIST.append(cls_name)

    def _get_img(self, img_id):
        img_desc = self.coco.imgs[img_id]
        img_fname = img_desc['file_name']
        img_fpath = os.path.join(self.img_dir, img_fname)
        return Image.open(img_fpath).convert('RGB')
    
    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        annotations = self.coco.imgToAnns[img_id]
        img = self._get_img(img_id)
        seg_mask = self._gen_seg_mask(annotations, img.size[1], img.size[0])
        return (img, seg_mask)
    
    def _gen_seg_mask(self, inst_annotations, h, w):
        # Note that this is different from implementation in those blogs from Google
        # Reference: https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/mscoco/segmentation.py
        mask = np.zeros((h, w), dtype=np.int64)
        for instance in inst_annotations:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            real_class_id = self.class_map[instance['category_id']]
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * real_class_id)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * real_class_id).astype(np.int64)
        return torch.tensor(mask)
    
    def get_class_map(self, class_id):
        return deepcopy((self.instance_class_map[class_id]))

    def __len__(self):
        return len(self.coco.imgs)
    
    def init_class_map(self):
        """
        init class map and write as txt to class_map_dir
        """
        instances_class_map = {}
        for c in range(1, 81):
            instances_class_map[c] = set() # avoid duplicated elements
        for i in range(len(self)):
            img_id = self.img_ids[i]
            annotations = self.coco.imgToAnns[img_id]
            for ann in annotations:
                if ann['iscrowd'] or ann['area'] < self.min_area:
                    continue # crowded examples are excluded
                real_class_id = self.class_map[ann['category_id']]
                instances_class_map[real_class_id].add(i)
        for c in range(1, 81):
            instances_class_map[c] = sorted(list(instances_class_map[c]))
        return instances_class_map

def get_train_set(cfg):
    ds = COCOSeg("/data/COCO2017/", True)
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    ds = COCOSeg("/data/COCO2017", False)
    return base_set(ds, "test", cfg)