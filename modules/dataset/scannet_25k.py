import os
import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image

import utils
from .baseset import base_set

class ScanNet25K(datasets.vision.VisionDataset):
    '''
    Semantic segmentation of ScanNet 25K downsampled data.

    Data availabel at http://kaldir.vc.in.tum.de/scannet/v2/tasks/scannet_frames_25k.zip
    '''

    # Only 20+1 classes are used
    # https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_label.py
    TRAIN_SPLIT_URL = "https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark/scannetv2_train.txt"
    VAL_SPLIT_URL = "https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark/scannetv2_val.txt"
    CLASS_NAMES_LIST = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',\
        'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet',\
            'sink', 'bathtub', 'otherfurniture']
    VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
    NYU_SCANNET_MAPPING = np.zeros(41, dtype=np.int) - 1 # NYU40 has 40 labels
    for i in range(len(VALID_CLASS_IDS)):
        NYU_SCANNET_MAPPING[VALID_CLASS_IDS[i]] = i

    def __init__(self, root, train=False):
        super(ScanNet25K, self).__init__(root, None, None, None)
        root = os.path.join(root, "scannet_frames_25k")

        train_split_path = os.path.join(root, "train_split.txt")
        val_split_path = os.path.join(root, "val_split.txt")

        if train:
            if not os.path.exists(train_split_path):
                utils.download_file(self.TRAIN_SPLIT_URL, train_split_path)
            split_path = train_split_path
        else:
            if not os.path.exists(val_split_path):
                utils.download_file(self.VAL_SPLIT_URL, val_split_path)
            split_path = val_split_path
        
        with open(split_path) as f:
            scene_list = [i.strip() for i in f.readlines()]

        self.img_paths = []
        self.target_paths = []

        for scene_name in scene_list:
            color_dir = os.path.join(root, scene_name, "color")
            label_dir = os.path.join(root, scene_name, "label")
            color_img_list = sorted(os.listdir(color_dir))
            label_img_list = sorted(os.listdir(label_dir))
            
            # color and label len should be equal
            assert len(color_img_list) == len(label_img_list), f"{scene_name} got {color_img_list} and {label_img_list}"

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
        target_np = self.NYU_SCANNET_MAPPING[target_np]

        return img, torch.tensor(target_np)

    def __len__(self):
        return len(self.img_paths)

def get_train_set(cfg):
    ds = ScanNet25K(utils.get_dataset_root(), train=True)
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    ds =ScanNet25K(utils.get_dataset_root(), train=False)
    return base_set(ds, "test", cfg)