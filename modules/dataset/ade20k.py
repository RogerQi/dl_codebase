import os
import json
from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms

import utils
from .baseset import base_set

class ADE20KSegReader(datasets.vision.VisionDataset):
    '''
    Coarse grain semantic segmentation dataset from the 2016 ADE20K challenge.

    Data can be grabbed from http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
    '''

    CLASS_NAMES_LIST = ["background", "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair", "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion", "base", "box", "column", "signboard", "chest", "counter", "sand", "sink", "skyscraper", "fireplace", "refrigerator", "grandstand", "path", "stairs", "runway", "case", "pool", "pillow", "screen", "stairway", "river", "bridge", "bookcase", "blind", "coffee", "toilet", "flower", "book", "hill", "bench", "countertop", "stove", "palm", "kitchen", "computer", "swivel", "boat", "bar", "arcade", "hovel", "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth", "television", "airplane", "dirt", "apparel", "pole", "land", "bannister", "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van", "ship", "fountain", "conveyer", "canopy", "washer", "plaything", "swimming", "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball", "food", "step", "tank", "trade", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic", "tray", "ashcan", "fan", "pier", "crt", "plate", "monitor", "bulletin", "shower", "radiator", "glass", "clock", "flag"]

    def __init__(self, root, train=True):
        '''
        Initialize and load the ADE20K annotation file into memory.
        '''
        super(ADE20KSegReader, self).__init__(root, None, None, None)
        self.train = train
        self.base_dir = os.path.join(root, "ade20k_coarse")

        if train:
            dataset_json_path = os.path.join(self.base_dir, "training.odgt")
        else:
            dataset_json_path = os.path.join(self.base_dir, "validation.odgt")

        self.ds = [json.loads(x.rstrip()) for x in open(dataset_json_path, 'r')]
    
    def __getitem__(self, idx: int):
        """
        Args:
            key (int): key

        Returns:
            ret_dict
        """
        img_path = os.path.join(self.base_dir, self.ds[idx]['fpath_img'])
        seg_path = os.path.join(self.base_dir, self.ds[idx]['fpath_segm'])
        raw_img = Image.open(img_path).convert('RGB')
        segm = Image.open(seg_path)
        assert(segm.mode == "L")
        assert(raw_img.size[0] == segm.size[0])
        assert(raw_img.size[1] == segm.size[1])
        seg_mask = torch.tensor(np.array(segm, dtype = np.uint8), dtype = torch.int64)
        # seg_mask = self.label_unifier(seg_mask)
        return (raw_img, seg_mask)

    def __len__(self):
        return len(self.ds)

def get_train_set(cfg):
    # Note: previous works including FCN (https://arxiv.org/pdf/1411.4038.pdf)
    # or OSLSM (https://arxiv.org/pdf/1709.03410.pdf) use SBD annotations.
    # Here we follow the convention and use augmented notations from SBD
    ds = ADE20KSegReader(utils.get_dataset_root(), True)
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    ds = ADE20KSegReader(utils.get_dataset_root(), False)
    return base_set(ds, "test", cfg)
