import sys
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from .baseset import base_set

toTensorFunc = transforms.ToTensor()
download_sbd = False if os.path.exists('/data/sbd') else True
download_voc = False if os.path.exists('/data/VOCdevkit') else True

class voc2012_seg(datasets.VOCSegmentation):
    def __getitem__(self, idx):
        img, label_pil = datasets.VOCSegmentation.__getitem__(self, idx)
        label_np = np.array(label_pil)
        # Only 20 classes in VOC2012. Those pixels whose class is
        # greater than 20 are edges
        label_np[label_np > 20] = 0
        return img, torch.tensor(label_np)

class sbd_seg(datasets.SBDataset):
    def __getitem__(self, idx):
        img, label_pil = datasets.SBDataset.__getitem__(self, idx)
        img = toTensorFunc(img)
        label_np = np.array(label_pil, dtype=np.long)
        return img, torch.tensor(label_np)

def get_train_set(cfg):
    # Note 1: previous works including FCN (https://arxiv.org/pdf/1411.4038.pdf)
    # or OSLSM (https://arxiv.org/pdf/1709.03410.pdf) use SBD annotations.
    # The Pascal VOC2012 challenge only has a small subset of images annotated
    # for semantic segmentation (~1400 in training set and ~1500 in validation set)
    # while SBD annotates ~11500 images (~8500 in training set and ~2900 in validation)

    # Note 2: for some reason, torchvision.datasets.SBDataset does not support transform
    # parameter (which transforms only the image but not the label mask). So we have to
    # manually implement this method.
    ds = sbd_seg('/data/sbd', image_set='train', mode="segmentation", download=download_sbd)
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    ds = voc2012_seg('/data/', image_set='val', download = download_voc, transform=transforms.ToTensor())
    if len(ds) == 1449:
        for _ in range(5):
            print("USING ORIGINAL VOC2012 VAL SET, WHICH INTERSECT WITH SBD")
            print("RUN utils_main/computer_sbd_voc_val_non_intersect.py TO AVOID THIS")
    return base_set(ds, "test", cfg)