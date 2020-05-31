import os
import json
import numpy as np
from scipy.io import loadmat
import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image
from .baseset import base_set

def get_seg_filename(img_name):
    return img_name[:-4] + "_seg.png"

def decode_class_id(r, g):
    return r // 10 * 256 + g

class ADE20K(datasets.vision.VisionDataset):
    '''
    Fine-grained instance-level segmentation data from the 2016 ADE20K challenge.

    Data can be grabbed from https://groups.csail.mit.edu/vision/datasets/ADE20K/
    '''
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        '''
        Initialize and load the ADE20K annotation file into memory.

        Args:
            - root: path to the folder containing the ADE20K_2016_07_26 folder.
                e.g. It should be /data if images are in /data/ADE20K_2016_07_26/images
            - annFile: path to the serialized Matlab array file provided in the dataset.
                e.g. /data/ADE20K_2016_07_26/index_ade20k.mat
        '''
        super(ADE20K, self).__init__(root, transforms, transform, target_transform)
        self.ds = loadmat(annFile)
        self.ds = self.ds["index"]
        assert self.ds['filename'][0, 0].shape == self.ds['folder'][0, 0].shape
        self.dataset_size = self.ds['filename'][0, 0].shape[1]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target), where target is a tensor of shape (H, W) and type torch.uint64.
                Each element is ranged between (0, num_classes - 1).
        """
        img_name = self.ds['filename'][0, 0][0, index][0]
        folder_path = self.ds['folder'][0, 0][0, index][0]
        img_path = os.path.join(self.root, folder_path, img_name)
        seg_path = os.path.join(self.root, folder_path, get_seg_filename(img_name))
        img = np.array(Image.open(img_path).convert('RGB'), dtype = np.uint8)
        raw_seg_mask = np.array(Image.open(seg_path), dtype = np.uint8)
        assert img.shape == raw_seg_mask.shape, 'Mismatched shape {0}/{1} for {2} and {3}'.format(img.shape, raw_seg_mask.shape, img_path, seg_path)
        H, W, C = raw_seg_mask.shape
        seg_mask = torch.zeros((H, W), dtype = torch.int64)
        if True:
            # Generate to H x W format with each element in (0, C - 1)
            seg_mask = raw_seg_mask[:,:,0] // 10
            seg_mask = seg_mask * 256
            seg_mask = seg_mask + raw_seg_mask[:,:,1]
        else:
            # Get C x H x W format with each element in (0, 1)
            raise NotImplementedError
        return (img, seg_mask)

    def __len__(self):
        return self.dataset_size
    
    def get_class_name(self, cls_id):
        return self.ds['objectnames'][0, 0][0, cls_id][0]

def get_train_set(cfg):
    ds = ADE20K(
        "/data/",
        "/data/ADE20K_2016_07_26/index_ade20k.mat",
        transform = transforms.ToTensor()
    )
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    return None
    ds = ADE20K(
        "/data/COCO2017/val2017/",
        "/data/COCO2017/annotations/panoptic_val2017.json",
        "/data/COCO2017/annotations/panoptic_semantic_val2017/",
        transform = transforms.ToTensor()
    )
    return base_set(ds, "test", cfg)