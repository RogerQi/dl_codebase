import os
import json
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image
from .baseset import base_set

# 2017 train images normalization constants
#   mean: 0.4700, 0.4468, 0.4076
#   sd: 0.2439, 0.2390, 0.2420

class CocoSemantic(datasets.vision.VisionDataset):
    '''
    Semantic segmentation of COCO2017 data generated from COCO 2017 Panoptic Annotation.

    Should you wish to use this, please refer to main/utils/coco_panoptic_2_seg.py to pre-process data.
    '''
    def __init__(self, root, annFile, semantic_seg_folder, transform=None, target_transform=None, transforms=None):
        super(CocoSemantic, self).__init__(root, transforms, transform, target_transform)
        with open(annFile, 'r') as f:
            self.coco_p = json.load(f)
        assert len(self.coco_p['annotations']) == len(self.coco_p['images'])
        self.dataset_size = len(self.coco_p['annotations'])
        self.semantic_seg_folder = semantic_seg_folder

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target), where target is a tensor of shape (H, W) and type torch.uint64.
                Each element is ranged between (0, num_classes - 1).
        """
        # Get image name and paths
        annotation = self.coco_p['annotations'][index]
        img_name = annotation['file_name']
        # Get raw image
        raw_img_name = img_name[:-3] + 'jpg'
        img_path = os.path.join(self.root, raw_img_name)
        img = np.array(Image.open(img_path).convert('RGB'), dtype = np.uint8)
        # Get annotation
        seg_img_path = os.path.join(self.semantic_seg_folder, img_name)
        target = torch.tensor(np.array(Image.open(seg_img_path), dtype=np.int64))

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return self.dataset_size

def get_train_set(cfg):
    ds = CocoSemantic(
        "../datasets/COCO2017/train2017/",
        "../datasets/COCO2017/annotations/panoptic_train2017.json",
        "/home/roger/datasets/COCO2017/annotations/panoptic_semantic_train2017/",
        transform = transforms.ToTensor()
    )
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    ds = CocoSemantic(
        "../datasets/COCO2017/val2017/",
        "../datasets/COCO2017/annotations/panoptic_val2017.json",
        "/home/roger/datasets/COCO2017/annotations/panoptic_semantic_val2017/",
        transform = transforms.ToTensor()
    )
    return base_set(ds, "test", cfg)