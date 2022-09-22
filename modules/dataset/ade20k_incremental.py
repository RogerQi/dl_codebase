import os
import numpy as np
import torch
from PIL import Image
from torchvision import datasets
from .baseset import base_set
from copy import deepcopy
import utils

from .ade20k import ADE20KSegReader

class ADE20KIncrementalReader(datasets.vision.VisionDataset):
    def __init__(self, root, split, exclude_novel=False, vanilla_label=False):
        """
        pascal_5i dataset reader

        Parameters:
            - root:  root to data folder containing SBD and VOC2012 dataset. See README.md for details
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
        super(ADE20KIncrementalReader, self).__init__(root, None, None, None)
        if vanilla_label:
            assert exclude_novel
        self.vanilla_label = vanilla_label

        # Get augmented VOC dataset
        self.vanilla_ds = ADE20KSegReader(root, split)
        self.CLASS_NAMES_LIST = self.vanilla_ds.CLASS_NAMES_LIST

        # Split dataset based on folding. Refer to https://arxiv.org/pdf/1709.03410.pdf
        # Given fold number, define L_{test}
        self.val_label_set = [i for i in range(101, 151)]
        self.train_label_set = [i for i in range(1, 101)]
        
        self.visible_labels = self.train_label_set
        self.invisible_labels = self.val_label_set
        
        # Pre-training or meta-training
        if exclude_novel:
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
        for c in range(1, 151):
            self.class_map[c] = []
            real_class_map = self.vanilla_ds.get_class_map(c)
            for subset_i, real_idx in enumerate(self.subset_idx):
                if real_idx in real_class_map:
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
        if not self.vanilla_label:
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
        min_val_label = min(self.val_label_set)
        max_val_label = max(self.val_label_set)
        greater_pixel_idx = (target_tensor > max_val_label)
        novel_pixel_idx = torch.logical_and(target_tensor >= min_val_label, torch.logical_not(greater_pixel_idx))
        target_tensor[novel_pixel_idx] = 0
        target_tensor[greater_pixel_idx] -= len(self.val_label_set)
        return target_tensor

def get_train_set(cfg):
    ds = ADE20KIncrementalReader(utils.get_dataset_root(), True, exclude_novel=False)
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    ds = ADE20KIncrementalReader(utils.get_dataset_root(), False, exclude_novel=False)
    return base_set(ds, "test", cfg)

def get_train_set_vanilla_label(cfg):
    ds = ADE20KIncrementalReader(utils.get_dataset_root(), True, exclude_novel=True, vanilla_label=True)
    return base_set(ds, "train", cfg)

def get_continual_train_set(cfg):
    ds = ADE20KSegReader(utils.get_dataset_root(), True)
    return base_set(ds, "train", cfg)

def get_continual_test_set(cfg):
    ds = ADE20KSegReader(utils.get_dataset_root(), False)
    return base_set(ds, "test", cfg)

