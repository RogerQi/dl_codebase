import os
import numpy as np
import torch
from PIL import Image
from copy import deepcopy
from torchvision import datasets
from .baseset import base_set

import utils

from IPython import embed

class LVIS150Reader(datasets.vision.VisionDataset):

    CLASS_NAMES_LIST = ['void', 'air_conditioner', 'awning', 'barrel', 'bath_towel', 'bell',
                        'birthday_cake', 'boat', 'bowl', 'bun', 'calendar',
                        'cauliflower', 'Christmas_tree', 'coffee_table', 'cover',
                        'cupboard', 'dishtowel', 'doughnut', 'earring', 'fire_engine',
                        'flower_arrangement', 'goggles', 'ham', 'helmet', 'jar',
                        'knob', 'latch', 'log', 'mast', 'motor', 'napkin', 'ottoman',
                        'parking_meter', 'pickle', 'plastic_bag', 'potato',
                        'refrigerator', 'saltshaker', 'scoreboard', 'shower_curtain',
                        'ski_pole', 'speaker_(stero_equipment)', 'stove', 'sunglasses',
                        'taillight', 'telephone_pole', 'toaster_oven', 'tote_bag',
                        'tray', 'vent', 'weathervane', 'aerosol_can', 'artichoke',
                        'barrette', 'beanbag', 'birdhouse', 'bottle_opener', 'bull',
                        'calf', 'cantaloup', 'cayenne_(spice)', 'cincture', 'cock',
                        'corset', 'crock_pot', 'dartboard', 'doormat', 'eggbeater',
                        'fire_hose', 'foal', 'frog', 'gift_wrap', 'grizzly',
                        'handcart', 'honey', 'ironing_board', 'kitten', 'lollipop',
                        'meatball', 'musical_instrument', 'orange_juice', 'palette',
                        'passport', 'pet', 'pocketknife', 'prawn', 'radish',
                        'record_player', 'salmon_(fish)', 'shaker', 'sled',
                        'spice_rack', 'step_stool', 'sweet_potato', 'teakettle',
                        'tights_(clothing)', 'trophy_cup', 'vacuum_cleaner',
                        'walking_stick', 'webcam', 'window_box_(for_plants)',
                        'applesauce', 'ax', 'beachball', 'bonnet', 'bowling_ball',
                        'bubble_gum', 'candy_bar', 'cassette', 'chocolate_mousse',
                        'comic_book', 'cornbread', 'cream_pitcher', 'cylinder', 'die',
                        'dragonfly', 'dustpan', 'egg_roll', 'ferret', 'funnel',
                        'garbage', 'gourd', 'hair_curler', 'handsaw', 'heron',
                        'hotplate', 'kitchen_table', 'lab_coat', 'leather', 'liquor',
                        'matchbox', 'milkshake', 'pan_(metal_container)',
                        'paperback_book', 'pencil_sharpener', 'phonebook',
                        'pirate_flag', 'poncho', 'puffin', 'road_map', 'salmon_(food)',
                        'scarecrow', 'shaver_(electric)', 'shot_glass', 'space_shuttle',
                        'stepladder', 'sugar_bowl', 'table-tennis_table', 'trampoline',
                        'tux', 'waffle_iron']
    
    def __init__(self, base_dir, train):
        super(LVIS150Reader, self).__init__(base_dir, None, None, None)
        assert train in [True, False]
        self.train = train
        self.base_dir = base_dir

        if self.train:
            metadata_path = 'metadata/lvis_150/train.txt'
        else:
            metadata_path = 'metadata/lvis_150/val.txt'
        
        with open(metadata_path) as f:
            self.img_info_list = f.read().splitlines()
        
        self.class_map = {}

        if self.train:
            class_map_dir = "metadata/lvis150/train"
        else:
            class_map_dir = "metadata/lvis150/val"

        for c in range(1, 151):
            class_map_path = os.path.join(class_map_dir, str(c) + ".txt")
            class_idx_list = list(np.loadtxt(class_map_path, dtype=int).reshape((-1,)))
            self.class_map[c] = class_idx_list
    
    def __len__(self):
        return len(self.img_info_list)
    
    def __getitem__(self, idx):
        # Load image
        img_id, split_folder, fn = self.img_info_list[idx].split(' ')
        
        img_path = os.path.join(self.base_dir, split_folder, fn)
        img_np = np.array(Image.open(img_path).convert('RGB'))
        
        mask_fn = fn.replace('.jpg', '.png')
        mask_path = os.path.join(self.base_dir, 'lvis_150_masks', split_folder, mask_fn)
        mask_np = np.array(Image.open(mask_path))
        
        return img_np, torch.tensor(mask_np).long()
    
    def get_label_range(self):
        return [i for i in range(1, 151)]
    
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

def get_train_set(cfg):
    COCO_PATH = os.path.join(utils.get_dataset_root(), "COCO2017")
    ds = LVIS150Reader(COCO_PATH, True)
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    COCO_PATH = os.path.join(utils.get_dataset_root(), "COCO2017")
    ds = LVIS150Reader(COCO_PATH, False)
    return base_set(ds, "test", cfg)
