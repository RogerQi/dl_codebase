import sys
import os
import json
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image

import utils
from .baseset import base_set

class ScanNet25K(datasets.vision.VisionDataset):
    '''
    Semantic segmentation of ScanNet 25K downsampled data.

    Data availabel at http://kaldir.vc.in.tum.de/scannet/v2/tasks/scannet_frames_25k.zip
    '''
    def __init__(self, root):
        super(ScanNet25K, self).__init__(root, None, None, None)
        root = os.path.join(root, "scannet_frames_25k")
        scene_list = sorted(os.listdir(root))

        self.img_paths = []
        self.target_paths = []

        for scene_name in scene_list:
            color_dir = os.path.join(root, scene_name, "color")
            label_dir = os.path.join(root, scene_name, "label")
            color_img_list = sorted(os.listdir(color_dir))
            label_img_list = sorted(os.listdir(label_dir))
            
            # color and label len should be equal
            assert len(color_img_list) == len(label_img_list)

            # assert frame numbers are the same. We can't use == directly because
            # RGB images are stored as .jpg while labels are stored as .png
            assert [i.split('.')[0] for i in color_img_list] == [i.split('.')[0] for i in label_img_list]
            
            color_img_paths = [os.path.join(root, scene_name, "color", fn) for fn in color_img_list]
            label_img_paths = [os.path.join(root, scene_name, "label", fn) for fn in label_img_list]

            self.img_paths = self.img_paths + color_img_paths
            self.target_paths = self.target_paths + label_img_paths

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target), where target is a tensor of shape (H, W) and type torch.uint64.
                Each element is ranged between (0, num_classes - 1).
        """
        # Get image name and paths
        img = Image.open(self.img_paths[idx]).convert("RGB")

        target = Image.open(self.target_paths[idx])
        target_np = np.array(target, dtype=np.long)

        return img, torch.tensor(target_np)

    def __len__(self):
        return len(self.img_paths)

def get_train_set(cfg):
    ds = ScanNet25K(utils.get_dataset_root())
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    ds =ScanNet25K(utils.get_dataset_root())
    return base_set(ds, "test", cfg)