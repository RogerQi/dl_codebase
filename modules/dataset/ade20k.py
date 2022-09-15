import os
import numpy as np
import torch
from PIL import Image
from copy import deepcopy
from torchvision import datasets
from .baseset import base_set

import utils

class ADE20KSegReader(datasets.vision.VisionDataset):

    CLASS_NAMES_LIST = [
        "void", "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed ", "windowpane",
        "grass", "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant",
        "curtain", "chair", "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror",
        "rug", "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub",
        "railing", "cushion", "base", "box", "column", "signboard", "chest of drawers", "counter",
        "sand", "sink", "skyscraper", "fireplace", "refrigerator", "grandstand", "path", "stairs",
        "runway", "case", "pool table", "pillow", "screen door", "stairway", "river", "bridge",
        "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench", "countertop",
        "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine",
        "hovel", "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight",
        "booth", "television receiver", "airplane", "dirt track", "apparel", "pole", "land",
        "bannister", "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van", "ship",
        "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool", "stool",
        "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball", "food",
        "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
        "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan",
        "fan", "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator",
        "glass", "clock", "flag"
    ]

    def __init__(self, root, train=True):
        super(ADE20KSegReader, self).__init__(root, None, None, None)

        self.train = train
        self.base_dir = os.path.join(root, "ADEChallengeData2016")

        if self.train:
            split = 'training'
        else:
            split = 'validation'

        annotation_folder = os.path.join(self.base_dir, 'annotations', split)
        image_folder = os.path.join(self.base_dir, 'images', split)

        fnames = sorted(os.listdir(image_folder))
        self.images = [os.path.join(image_folder, x) for x in fnames]
        self.masks = [os.path.join(annotation_folder, x[:-3] + "png") for x in fnames]

        assert len(self.images) == len(self.masks)

        self.class_map = {}

        if self.train:
            class_map_dir = "metadata/ade20k/train"
        else:
            class_map_dir = "metadata/ade20k/val"

        for c in range(1, 151):
            class_map_path = os.path.join(class_map_dir, str(c) + ".txt")
            class_idx_list = list(np.loadtxt(class_map_path, dtype=int))
            self.class_map[c] = class_idx_list

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = np.array(Image.open(self.masks[index])).astype(np.int)

        target[target == 255] = -1

        return img, torch.tensor(target).long()

    def __len__(self):
        return len(self.images)
    
    def get_class_map(self, class_id):
        """
        Given a class label id (e.g., 2), return a list of all images in
        the dataset containing at least one pixel of the class.

        Parameters:
            - class_id: an integer representing class

        Return:
            - a list of all images in the dataset containing at least one pixel of the class
        """
        return deepcopy(self.class_map[class_id])
    
    def get_label_range(self):
        return [i for i in range(1, 151)]

def get_train_set(cfg):
    ds = ADE20KSegReader(utils.get_dataset_root(), True)
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    ds = ADE20KSegReader(utils.get_dataset_root(), False)
    return base_set(ds, "test", cfg)
