import sys
import os
import json
import numpy as np
import torch
import torchvision
import shutil

from copy import deepcopy
from torchvision import datasets, transforms
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from .baseset import base_set
from tqdm import trange

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
        
        # Given a class idx (1-80), self.instance_class_map gives the list of images that contain
        # this class idx
        class_map_dir = os.path.join(root, 'instance_seg_class_map', split_name)
        self.mask_dir = os.path.join(root, 'annotations', 'instance_semantic_mask_{}'.format(split_name))
        if not os.path.exists(class_map_dir):
            # Merge VOC and SBD datasets and create auxiliary files
            try:
                assert not os.path.exists(self.mask_dir)
                self.create_coco_class_map(class_map_dir, self.mask_dir)
            except (Exception, KeyboardInterrupt) as e:
                # Dataset creation fail for some reason...
                shutil.rmtree(class_map_dir)
                shutil.rmtree(self.mask_dir)
                raise e
        
        self.instance_class_map = {}
        for c in range(1, 81):
            class_map_path = os.path.join(class_map_dir, str(c) + ".txt")
            class_idx_list = list(np.loadtxt(class_map_path, dtype='str'))
            # Map name to indices
            class_idx_list = [int(i) for i in class_idx_list]
            self.instance_class_map[c] = class_idx_list

        self.CLASS_NAMES_LIST = ['background']
        for i in range(len(class_list)):
            cls_name = self.coco.cats[class_list[i]]['name']
            self.CLASS_NAMES_LIST.append(cls_name)
    
    def create_coco_class_map(self, class_map_dir, mask_dir):
        assert not os.path.exists(class_map_dir)
        assert not os.path.exists(mask_dir)
        os.makedirs(class_map_dir)
        os.makedirs(mask_dir)

        instance_class_map = {}
        for c in range(1, 81):
            instance_class_map[c] = []

        for i in trange(len(self)):
            img_id = self.img_ids[i]
            mask = self._get_mask(img_id)
            contained_labels = torch.unique(mask)
            for c in contained_labels:
                c = int(c)
                if c == 0 or c == -1:
                    continue # background or ignore_mask
                instance_class_map[c].append(str(i))
            serialized_path = os.path.join(mask_dir, '{}.npy'.format(img_id))
            with open(serialized_path, 'wb') as f:
                np.save(f, mask.numpy())
        
        for c in range(1, 81):
            with open(os.path.join(class_map_dir, str(c) + '.txt'), 'w') as f:
                f.write('\n'.join(instance_class_map[c]))

    def _get_img(self, img_id):
        img_desc = self.coco.imgs[img_id]
        img_fname = img_desc['file_name']
        img_fpath = os.path.join(self.img_dir, img_fname)
        return Image.open(img_fpath).convert('RGB')
    
    def _get_mask(self, img_id):
        img_metadata = self.coco.loadImgs(img_id)[0]
        annotations = self.coco.imgToAnns[img_id]
        seg_mask = self._gen_seg_mask(annotations, img_metadata['height'], img_metadata['width'])
        return seg_mask
    
    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        img = self._get_img(img_id)
        mask_path = os.path.join(self.mask_dir, '{}.npy'.format(img_id))
        seg_mask = np.load(mask_path)
        return (img, torch.tensor(seg_mask))
    
    def _gen_seg_mask(self, annotations, h, w):
        seg_mask = torch.zeros((h, w), dtype=torch.int64)
        for ann in annotations:
            real_class_id = self.class_map[ann['category_id']]
            ann_mask = torch.from_numpy(self.coco.annToMask(ann))
            # mask indicating invalid regions
            if ann['iscrowd'] or ann['area'] < self.min_area:
                seg_mask[ann_mask > 0] = -1
            else:
                assert real_class_id >= 0 and real_class_id <= 80
                seg_mask = torch.max(seg_mask, ann_mask * real_class_id)
        return seg_mask.long()
    
    def get_class_map(self, class_id):
        return deepcopy((self.instance_class_map[class_id]))

    def __len__(self):
        return len(self.coco.imgs)

def get_train_set(cfg):
    ds = COCOSeg("/data/COCO2017/", True)
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    ds = COCOSeg("/data/COCO2017", False)
    return base_set(ds, "test", cfg)