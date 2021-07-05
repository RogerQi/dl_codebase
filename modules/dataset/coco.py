import sys
import os
import json
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image
from pycocotools.coco import COCO
from .baseset import base_set

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
        self.class_map = {}
        for i in range(len(class_list)):
            self.class_map[class_list[i]] = i + 1

    def _get_img(self, img_id):
        img_desc = self.coco.imgs[img_id]
        img_fname = img_desc['file_name']
        img_fpath = os.path.join(self.img_dir, img_fname)
        return Image.open(img_fpath).convert('RGB')
    
    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        annotations = self.coco.imgToAnns[img_id]
        img = self._get_img(img_id)
        seg_mask = torch.zeros((img.size[1], img.size[0]), dtype=torch.int64)
        for ann in annotations:
            real_class_id = self.class_map[ann['category_id']]
            ann_mask = torch.from_numpy(self.coco.annToMask(ann))
            # mask indicating invalid regions
            if ann['iscrowd'] or ann['area'] < self.min_area:
                seg_mask[ann_mask > 0] = -1
            else:
                assert real_class_id >= 0 and real_class_id <= 80
                seg_mask = torch.max(seg_mask, ann_mask * real_class_id)
        return (img, seg_mask.long())

    def __len__(self):
        return len(self.coco.imgs)

def get_train_set(cfg):
    ds = COCOSeg("/data/COCO2017/", True)
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    ds = COCOSeg("/data/COCO2017", False)
    return base_set(ds, "test", cfg)