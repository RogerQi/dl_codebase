import os
import json
import pickle
import numpy as np
from copy import deepcopy
from PIL import Image

import utils
from .baseset import base_set


class mini_background_img:
    def __init__(self, root, split, bg_label=-100):
        assert split in ["base", "val", "novel"]
        if split == "base":
            pickle_path = os.path.join(root, "mini_imagenet", "background.pkl")
            assert bg_label == 64
        elif split == "val":
            pickle_path = os.path.join(root, "mini_imagenet", "background.pkl")
            assert bg_label == 64 + 16
        else:
            pickle_path = os.path.join(root, "mini_imagenet", "novel_background.pkl")
            assert bg_label == 64 + 16 + 20
        with open(pickle_path, 'rb') as f:
            self.img_paths_list = pickle.load(f)
        self.bg_label = bg_label
    
    def __len__(self):
        return len(self.img_paths_list)

    def __getitem__(self, idx):
        img_path = self.img_paths_list[idx]
        img = Image.open(img_path).convert('RGB')
        return (img, self.bg_label) # -100 as pseudo label

class mini_imagenet_w_bg:
    def __init__(self, root, split):
        assert split in ["base", "val", "novel"]
        json_path = os.path.join(root, "mini_imagenet", "{}.json".format(split))
        with open(json_path, 'r') as f:
            self.json_data = json.load(f)
        assert len(self.json_data['image_labels']) == len(self.json_data['image_names'])
        self.label_range = list(np.unique(self.json_data['image_labels']))
        self.class_map = {}
        for i in range(len(self.json_data['image_labels'])):
            cur_label = self.json_data['image_labels'][i]
            if cur_label in self.class_map:
                self.class_map[cur_label].append(i)
            else:
                self.class_map[cur_label] = [i]
        max_label = np.max(self.json_data['image_labels'])
        self.bg_ds = mini_background_img(root, split, max_label + 1)
    
    def __len__(self):
        return len(self.json_data['image_labels']) + len(self.bg_ds)
    
    def __getitem__(self, idx):
        if idx < len(self.json_data['image_labels']):
            img_path = self.json_data['image_names'][idx]
            label = self.json_data['image_labels'][idx]
            img = Image.open(img_path).convert('RGB')
            return (img, label)
        else:
            idx = idx - len(self.json_data['image_labels'])
            return self.bg_ds[idx]
    
    def get_class_map(self, class_id):
        return deepcopy(self.class_map[class_id])

    def get_label_range(self):
        return deepcopy(self.label_range)

def get_train_set(cfg):
    ds = mini_imagenet_w_bg(utils.get_dataset_root(), 'base')
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    ds = mini_imagenet_w_bg(utils.get_dataset_root(), 'val')
    return base_set(ds, "test", cfg)

def get_test_set(cfg):
    ds = mini_imagenet_w_bg(utils.get_dataset_root(), 'novel')
    return base_set(ds, "test", cfg)

