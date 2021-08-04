"""
Module containing reader to parse pascal_5i dataset from SBD and VOC2012
"""
import os
import random
from PIL import Image
from scipy.io import loadmat
from copy import deepcopy
import numpy as np
import torch
import torchvision
from .baseset import base_set
from .coco import COCOSeg

from IPython import embed

class COCOGIFSReader(torchvision.datasets.vision.VisionDataset):
    # The authors of GIFS used classes that did not appear in ImageNet.
    # Refer to Section 5.1 in https://arxiv.org/pdf/2012.01415.pdf for details
    NOVEL_CLASSES_LIST = ["cow", "giraffe", "suitcase", "frisbee", "skateboard", "carrot", "scissors"]
    def __init__(self, root, base_stage, split, exclude_novel=False):
        """
        pascal_5i dataset reader

        Parameters:
            - root:  root to data folder containing SBD and VOC2012 dataset. See README.md for details
            - fold:  folding index as in OSLSM (https://arxiv.org/pdf/1709.03410.pdf)
            - base_stage: a bool flag to indicate whether L_{train} or L_{test} should be used
            - split: Specify train/val split of VOC2012 dataset to read from. True indicates training
            - exclude_novel: boolean flag to indicate whether novel examples are removed or masked.
                There are two cases:
                    * If set to True, examples containing pixels of novel classes are excluded.
                        (Generalized FS seg uses this setting)
                    * If set to False, examples containing pixels of novel classes are included.
                        Novel pixels will be masked as background. (FS seg uses this setting)
                When train=False (i.e., evaluating), this flag is ignored: images containing novel
                examples are always selected.
        """
        super(COCOGIFSReader, self).__init__(root, None, None, None)
        assert split in [True, False]
        self.base_stage = base_stage

        # Get augmented VOC dataset
        self.vanilla_ds = COCOSeg(root, split)
        self.CLASS_NAMES_LIST = self.vanilla_ds.CLASS_NAMES_LIST

        # Split dataset using classes not in ImageNet.
        self.val_label_set = sorted([self.CLASS_NAMES_LIST.index(n) for n in self.NOVEL_CLASSES_LIST])
        self.train_label_set = [i for i in range(
            1, 81) if i not in self.val_label_set]
        
        if self.base_stage:
            self.visible_labels = self.train_label_set
            self.invisible_labels = self.val_label_set
        else:
            self.visible_labels = self.val_label_set
            self.invisible_labels = self.train_label_set
        
        # Pre-training or meta-training
        if exclude_novel and self.base_stage:
            # Exclude images containing invisible classes and use rest
            novel_examples_list = []
            for label in self.invisible_labels:
                novel_examples_list += self.vanilla_ds.get_class_map(label)
            self.subset_idx = [i for i in range(len(self.vanilla_ds))]
            self.subset_idx = list(set(self.subset_idx) - set(novel_examples_list))
        else:
            # Use images containing at least one pixel from relevant classes
            examples_list = []
            for label in self.visible_labels:
                examples_list += self.vanilla_ds.get_class_map(label)
            self.subset_idx = list(set(examples_list))

        # Sort subset idx to make dataset deterministic (because set is unordered)
        self.subset_idx = sorted(self.subset_idx)

        # Generate self.class_map
        self.class_map = {}
        for c in range(1, 81):
            self.class_map[c] = []
            real_class_map = self.vanilla_ds.get_class_map(c)
            real_class_map_lut = {}
            for idx in real_class_map:
                real_class_map_lut[idx] = True
            for subset_i, real_idx in enumerate(self.subset_idx):
                if real_idx in real_class_map_lut:
                    self.class_map[c].append(subset_i)
    
    def __len__(self):
        return len(self.subset_idx)
    
    def get_class_map(self, class_id):
        """
        class_id here is subsetted. (e.g., class_idx is 12 in vanilla dataset may get translated to 2)
        """
        assert class_id > 0
        assert class_id < (len(self.visible_labels) + 1)
        # To access visible_labels, we translate class_id back to 0-indexed
        return deepcopy(self.class_map[self.visible_labels[class_id - 1]])
    
    def get_label_range(self):
        return [i + 1 for i in range(len(self.visible_labels))]

    def __getitem__(self, idx: int):
        assert 0 <= idx and idx < len(self.subset_idx)
        img, target_tensor = self.vanilla_ds[self.subset_idx[idx]]
        target_tensor = self.mask_pixel(target_tensor)
        return img, target_tensor

    def mask_pixel(self, target_tensor):
        """
        Following OSLSM, we mask pixels not in current label set as 0. e.g., when
        self.train = True, pixels whose labels are in L_{test} are masked as background

        Parameters:
            - target_tensor: segmentation mask (usually returned array from self.load_seg_mask)

        Return:
            - Offseted and masked segmentation mask
        """
        # Use the property that validation label split is contiguous to accelerate
        if self.base_stage:
            class_shift_factor = 0
            for novel_c in self.val_label_set:
                novel_c -= class_shift_factor
                target_tensor[target_tensor == novel_c] = 0
                target_tensor[target_tensor > novel_c] -= 1
                class_shift_factor += 1
            return target_tensor
        else:
            new_tensor = torch.zeros_like(target_tensor)
            for idx, novel_c in enumerate(self.val_label_set):
                new_tensor[target_tensor == novel_c] = idx + 1
            new_tensor[target_tensor == -1] = -1 # ignore mask
            return new_tensor

def get_train_set(cfg):
    ds = COCOGIFSReader("/data/COCO2017/", True, True, exclude_novel=True)
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    ds = COCOGIFSReader("/data/COCO2017/", True, False, exclude_novel=False)
    return base_set(ds, "test", cfg)

def get_meta_train_set(cfg):
    ds = COCOGIFSReader("/data/COCO2017/", True, True, exclude_novel=False)
    return base_set(ds, "train", cfg)

def get_meta_test_set(cfg):
    ds = COCOGIFSReader("/data/COCO2017/", False, False, exclude_novel=False)
    return base_set(ds, "test", cfg)

def get_continual_vanilla_train_set(cfg):
    ds = COCOSeg("/data/COCO2017/", True)
    return base_set(ds, "test", cfg) # Use test config to keep original scale of the image.

def get_continual_aug_train_set(cfg):
    ds = COCOSeg("/data/COCO2017/", True)
    return base_set(ds, "train", cfg)

def get_continual_test_set(cfg):
    ds = COCOSeg("/data/COCO2017/", False)
    return base_set(ds, "test", cfg)
