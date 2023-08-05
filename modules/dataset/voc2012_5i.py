import os
import shutil
from PIL import Image
from scipy.io import loadmat
from copy import deepcopy
import numpy as np
import torch
from torchvision import datasets, transforms
import torchvision

import utils
from .baseset import base_set

# tasks_voc = {
#     "offline": {
#         0: list(range(21)),
#     },
#     "19-1": {
#         0: list(range(20)),
#         1: [20],
#     },
#     "15-5": {
#         0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
#         1: [16, 17, 18, 19, 20]
#     },
#     "15-1":
#         {
#             0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
#             1: [16],
#             2: [17],
#             3: [18],
#             4: [19],
#             5: [20]
#         },
#     "10-1":
#         {
#             0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#             1: [11],
#             2: [12],
#             3: [13],
#             4: [14],
#             5: [15],
#             6: [16],
#             7: [17],
#             8: [18],
#             9: [19],
#             10: [20]
#         },
#     "5-5": {
#         0: [0, 1, 2, 3, 4, 5],
#         1: [6, 7, 8, 9, 10],
#         2: [11, 12, 13, 14, 15],
#         3: [16, 17, 18, 19, 20]
#     },
#     "5-3": {
#         0: [0, 1, 2, 3, 4, 5],
#         1: [6, 7, 8],
#         2: [9, 10, 11],
#         3: [12, 13, 14],
#         4: [15, 16, 17],
#         5: [18, 19, 20],
#     },
#     "5-1": {
#         0 : [0, 1, 2, 3, 4, 5],
#         1 : [6, ],
#         2 : [7, ],
#         3 : [8, ],
#         4 : [9, ],
#         5 : [10, ],
#         6 : [11, ],
#         7 : [12, ],
#         8 : [13, ],
#         9 : [14, ],
#         10: [15, ],
#         11: [16, ],
#         12: [17, ],
#         13: [18, ],
#         14: [19, ],
#         15: [20, ],
#     },
#     "2-2": {
#         0 : [0, 1, 2],
#         1 : [3, 4],
#         2 : [5, 6],
#         3 : [7, 8],
#         4 : [9, 10],
#         5 : [11, 12],
#         6 : [13, 14],
#         7 : [15, 16],
#         8 : [17, 18],
#         9 : [19, 20],
#     },
#     "2-1":{
#         0 : [0, 1, 2],
#         1 : [3, ],
#         2 : [4, ],
#         3 : [5, ],
#         4 : [6, ],
#         5 : [7, ],
#         6 : [8, ],
#         7 : [9, ],
#         8 : [10, ],
#         9 : [11, ],
#         10: [12, ],
#         11: [13, ],
#         12: [14, ],
#         13: [15, ],
#         14: [16, ],
#         15: [17, ],
#         16: [18, ],
#         17: [19, ],
#         18: [20, ],
#     },
#     "15-1_b":{
#         0: [0, 12, 9, 20, 7, 15, 8, 14, 16, 5, 19, 4, 1, 13, 2, 11],
#         1: [17], 2: [3], 3: [6], 4: [18], 5: [10]
#     },
#     "15-1_c":{
#         0: [0, 13, 19, 15, 17, 9, 8, 5, 20, 4, 3, 10, 11, 18, 16, 7],
#         1: [12], 2: [14], 3: [6], 4: [1], 5: [2]
#     },
#     "15-1_d":{
#         0: [0, 15, 3, 2, 12, 14, 18, 20, 16, 11, 1, 19, 8, 10, 7, 17],
#         1: [6], 2: [5], 3: [13], 4: [9], 5: [4]
#     },
#     "15-1_e":{
#         0: [0, 7, 5, 3, 9, 13, 12, 14, 19, 10, 2, 1, 4, 16, 8, 17],
#         1: [15], 2: [18], 3: [6], 4: [11], 5: [20]
#     },
#     "15-1_f":{
#         0: [0, 7, 13, 5, 11, 9, 2, 15, 12, 14, 3, 20, 1, 16, 4, 18],
#         1: [8], 2: [6], 3: [10], 4: [19], 5: [17]
#     },
#     "15-1_g":{
#         0: [0, 7, 5, 9, 1, 15, 18, 14, 3, 20, 10, 4, 19, 11, 17, 16],
#         1: [12], 2: [8], 3: [6], 4: [2], 5: [13]
#     },
#     "15-1_h":{
#         0: [0, 12, 9, 19, 6, 4, 10, 5, 18, 14, 15, 16, 3, 8, 7, 11],
#         1: [13], 2: [2], 3: [20], 4: [17], 5: [1]
#     },
#     "15-1_i":{
#         0: [0, 13, 10, 15, 8, 7, 19, 4, 3, 16, 12, 14, 11, 5, 20, 6],
#         1: [2], 2: [18], 3: [9], 4: [17], 5: [1]
#     },
#     "15-1_j":{
#         0: [0, 1, 14, 9, 5, 2, 15, 8, 20, 6, 16, 18, 7, 11, 10, 19],
#         1: [3], 2: [4], 3: [17], 4: [12], 5: [13]
#     },
#     "15-1_k":{
#         0: [0, 16, 13, 1, 11, 12, 18, 6, 14, 5, 3, 7, 9, 20, 19, 15],
#         1: [4], 2: [2], 3: [10], 4: [8], 5: [17]
#     },
#     "15-1_l":{
#         0: [0, 10, 7, 6, 19, 16, 8, 17, 1, 14, 4, 9, 3, 15, 11, 12],
#         1: [2], 2: [18], 3: [20], 4: [13], 5: [5]
#     },
#     "15-1_m":{
#         0: [0, 18, 4, 14, 17, 12, 10, 7, 3, 9, 1, 8, 15, 6, 13, 2],
#         1: [5], 2: [11], 3: [20], 4: [16], 5: [19]
#     },
#     "15-1_n":{
#         0: [0, 5, 4, 13, 18, 14, 10, 19, 15, 7, 9, 3, 2, 8, 16, 20],
#         1: [1], 2: [12], 3: [11], 4: [6], 5: [17]
#     },
#     "15-1_o":{
#         0: [0, 9, 12, 13, 18, 7, 1, 15, 17, 10, 8, 4, 5, 20, 16, 6],
#         1: [14], 2: [19], 3: [11], 4: [2], 5: [3]
#     },
#     "15-1_p":{
#         0: [0, 9, 12, 13, 18, 2, 11, 15, 17, 10, 8, 4, 5, 20, 16, 6],
#         1: [14], 2: [19], 3: [1], 4: [7], 5: [3]
#     },
#     "15-1_q":{
#         0: [0, 3, 14, 13, 18, 2, 11, 15, 17, 10, 8, 4, 5, 20, 16, 6],
#         1: [12], 2: [19], 3: [1], 4: [7], 5: [9]
#     },
#     "15-1_r":{
#         0: [0, 3, 14, 13, 1, 2, 11, 15, 17, 7, 8, 4, 5, 9, 16, 19],
#         1: [12], 2: [6], 3: [18], 4: [10], 5: [20]
#     },
#     "15-1_s":{
#         0: [0, 3, 14, 6, 1, 2, 11, 12, 17, 7, 20, 4, 5, 9, 16, 19],
#         1: [15], 2: [13], 3: [18], 4: [10], 5: [8]
#     },
#     "15-1_t":{
#         0: [0, 3, 15, 13, 1, 2, 11, 18, 17, 7, 20, 8, 5, 9, 16, 19],
#         1: [14], 2: [6], 3: [12], 4: [10], 5: [4]
#     },
#     "15-1_u":{
#         0: [0, 3, 15, 13, 14, 6, 11, 18, 17, 7, 20, 8, 4, 9, 16, 10],
#         1: [1], 2: [2], 3: [12], 4: [19], 5: [5]
#     },
#     "15-1_v":{
#         0: [0, 1, 2, 12, 14, 6, 19, 18, 17, 5, 20, 8, 4, 9, 16, 10],
#         1: [3], 2: [15], 3: [13], 4: [11], 5: [7]
#     },
#     "15-1_w":{
#         0: [0, 1, 2, 12, 14, 13, 19, 18, 7, 11, 20, 8, 4, 9, 16, 10],
#         1: [3], 2: [15], 3: [6], 4: [5], 5: [17]
#     },
# }


# def get_tasks(dataset, task, step=None):
#     if dataset == 'voc':
#         tasks = tasks_voc
#     else:
#         NotImplementedError
#
#     if step is None:
#         return tasks[task].copy()
#
#     return tasks[task][step].copy()


def get_dataset_list(mode, overlap=True):
    all_dataset = open(f"metadata/voc_{mode}_cls.txt",
                       "r").read().splitlines()

    # target_cls = get_tasks('voc', task, step)
    target_cls = list(range(21))

    if 0 in target_cls:
        target_cls.remove(0)

    dataset_list = []

    if overlap:
        fil = lambda c: any(x in target_cls for x in classes)
    else:
        target_cls_old = list(range(1, target_cls[0]))
        target_cls_cum = target_cls + target_cls_old + [0, 255]

        fil = lambda c: any(x in target_cls for x in classes) and all(
            x in target_cls_cum for x in c)

    for idx, classes in enumerate(all_dataset):
        str_split = classes.split(" ")

        img_name = str_split[0]
        classes = [int(s) + 1 for s in str_split[1:]]

        if fil(classes):
            dataset_list.append(img_name)

    return dataset_list



def maybe_download_voc(root):
    '''
    Helper function to download the Pascal VOC dataset
    '''
    # Use torchvision download routine to download dataset to root
    from torchvision import datasets
    if not os.path.exists(os.path.join(root, 'VOCdevkit')):
        voc_set = datasets.VOCSegmentation(root, image_set='train', download=True)


def load_seg_mask(file_path):
    """
    Load seg_mask from file_path (supports .mat and .png).

    Target masks in SBD are stored as matlab .mat; while those in VOC2012 are .png

    Parameters:
        - file_path: path to the segmenation file

    Return: a numpy array of dtype long and element range(0, 21) containing segmentation mask
    """
    if file_path.endswith('.mat'):
        mat = loadmat(file_path)
        target = Image.fromarray(mat['GTcls'][0]['Segmentation'][0])
    else:
        target = Image.open(file_path)
    target_np = np.array(target, dtype=np.int16)

    # in VOC, 255 is used as ignore_mask in many works
    target_np[target_np > 20] = -1
    return target_np


def create_pascal_voc_aug(root):
    voc_base = os.path.join(root, 'VOCdevkit', 'VOC2012')
    # voc_base = os.path.join(root, 'PascalVOC12')

    # Define path to relevant txt files
    # voc_train_list_path = os.path.join(
    #     voc_base, 'ImageSets', 'Segmentation', 'train.txt')
    voc_val_list_path = os.path.join(
        voc_base, 'ImageSets', 'Segmentation', 'val.txt')

    # Use np.loadtxt to load all train/val sets
    # voc_train_set = set(np.loadtxt(voc_train_list_path, dtype="str"))
    voc_val_set = set(np.loadtxt(voc_val_list_path, dtype="str"))

    ##############################################3
    voc_train_set = get_dataset_list("train", overlap=True)
    ##################################################

    trainaug_set = voc_train_set  # - voc_val_set
    val_set = voc_val_set

    new_base_dir = os.path.join(root, 'PASCAL_VOC12')
    assert not os.path.exists(new_base_dir)
    os.makedirs(new_base_dir)

    new_img_dir = os.path.join(new_base_dir, "raw_images")
    new_ann_dir = os.path.join(new_base_dir, "annotations")
    os.makedirs(new_img_dir)
    os.makedirs(new_ann_dir)

    def merge_and_save(name_list, class_map_dir, split_name):
        class_map = {}
        # There are 20 (foreground) classes in VOC Seg
        for c in range(1, 21):
            class_map[c] = []

        for name in name_list:
            image_path = os.path.join(voc_base, 'JPEGImages', name + '.jpg')
            # ann_path = os.path.join(voc_base, 'SegmentationClass', name + '.png')
            ann_path = os.path.join(voc_base, 'SegmentationClassAug', name + '.png')

            new_img_path = os.path.join(new_img_dir, name + '.jpg')
            new_ann_path = os.path.join(new_ann_dir, name + '.npy')
            shutil.copyfile(image_path, new_img_path)
            seg_mask_np = load_seg_mask(ann_path)

            for c in range(1, 21):
                if c in seg_mask_np:
                    class_map[c].append(name)

            with open(new_ann_path, 'wb') as f:
                np.save(f, seg_mask_np)

        class_map_dir = os.path.join(new_base_dir, class_map_dir)
        assert not os.path.exists(class_map_dir)
        os.makedirs(class_map_dir)
        for c in range(1, 21):
            with open(os.path.join(class_map_dir, str(c) + '.txt'), 'w') as f:
                f.write('\n'.join(class_map[c]))

        # Save set pointers
        with open(os.path.join(new_base_dir, split_name + '.txt'), 'w') as f:
            f.write('\n'.join(name_list))

    merge_and_save(trainaug_set, os.path.join(new_base_dir, "trainaug_class_map"), 'trainaug')
    merge_and_save(val_set, os.path.join(new_base_dir, "val_class_map"), 'val')


class VOC12SegReader(datasets.vision.VisionDataset):
    """
    pascal_5i dataset reader

    Parameters:
        - root:  root to data folder containing SBD and VOC2012 dataset. See README.md for details
        - fold:  folding index as in OSLSM (https://arxiv.org/pdf/1709.03410.pdf)
        - train: a bool flag to indicate whether L_{train} or L_{test} should be used
    """

    CLASS_NAMES_LIST = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "potted plant",
        "sheep",
        "sofa",
        "train",
        "tv/monitor"
    ]

    def __init__(self, root, train, download=True):
        super(VOC12SegReader, self).__init__(root, None, None, None)
        self.train = train

        if download:
            maybe_download_voc(root)

        base_dir = os.path.join(root, "PASCAL_VOC12")
        if not os.path.exists(base_dir):
            # Merge VOC and SBD datasets and create auxiliary files
            try:
                create_pascal_voc_aug(root)
            except (Exception, KeyboardInterrupt) as e:
                # Dataset creation fail for some reason...
                shutil.rmtree(base_dir)
                raise e

        if train:
            name_path = os.path.join(base_dir, 'trainaug.txt')
            class_map_dir = os.path.join(base_dir, 'trainaug_class_map')
        else:
            name_path = os.path.join(base_dir, 'val.txt')
            class_map_dir = os.path.join(base_dir, 'val_class_map')

        # Read files
        name_list = list(np.loadtxt(name_path, dtype='str'))
        self.images = [os.path.join(base_dir, "raw_images", n + ".jpg") for n in name_list]
        self.targets = [os.path.join(base_dir, "annotations", n + ".npy") for n in name_list]

        # Given a class idx (1-20), self.class_map gives the list of images that contain
        # this class idx
        self.class_map = {}
        for c in range(1, 21):
            class_map_path = os.path.join(class_map_dir, str(c) + ".txt")
            class_name_list = list(np.loadtxt(class_map_path, dtype='str'))
            # Map name to indices
            class_idx_list = [name_list.index(n) for n in class_name_list]
            self.class_map[c] = class_idx_list

        # print("Size of image dataset after set is: " + str(len(self.images)))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")

        target_np = np.load(self.targets[index])

        return img, torch.tensor(target_np).long()

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
        return [i + 1 for i in range(20)]




class VOC125iReader(torchvision.datasets.vision.VisionDataset):
    def __init__(self, root, fold, base_stage, split, exclude_novel=False, vanilla_label=False):
        """
        pascal_5i dataset reader

        Parameters:
            - root:  root to data folder containing SBD and VOC2012 dataset. See README.md for details
            - fold:  folding index as in OSLSM (https://arxiv.org/pdf/1709.03410.pdf)
            - base_stage: a bool flag to indicate whether L_{train} or L_{test} should be used
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
        super(VOC125iReader, self).__init__(root, None, None, None)
        assert fold >= 0 and fold <= 3
        assert base_stage
        if vanilla_label:
            assert exclude_novel
        self.vanilla_label = vanilla_label
        self.base_stage = base_stage

        # Get augmented VOC dataset
        self.vanilla_ds = VOC12SegReader(root, split, download=True)
        self.CLASS_NAMES_LIST = self.vanilla_ds.CLASS_NAMES_LIST

        # Split dataset based on folding. Refer to https://arxiv.org/pdf/1709.03410.pdf
        # Given fold number, define L_{test}
        self.val_label_set = [i for i in range(fold * 5 + 1, fold * 5 + 6)]
        self.train_label_set = [i for i in range(
            1, 21) if i not in self.val_label_set]

        if self.base_stage:
            self.visible_labels = self.train_label_set
            self.invisible_labels = self.val_label_set
        else:
            self.visible_labels = self.val_label_set
            self.invisible_labels = self.train_label_set

        # Pre-training or meta-training
        if exclude_novel and self.base_stage:
            # Exclude images containing invisible classes and use rest
            novel_examples_list = []
            for label in self.invisible_labels:
                novel_examples_list += self.vanilla_ds.get_class_map(label)
            self.subset_idx = [i for i in range(len(self.vanilla_ds))]
            self.subset_idx = list(set(self.subset_idx) - set(novel_examples_list))
            ###############################################################################
            # also include those images containing invisible classes
            # self.subset_idx = list(set(self.subset_idx))
            ###############################################################################
        else:
            # Use images containing at least one pixel from relevant classes
            examples_list = []
            #################################################################
            for label in self.visible_labels:
                examples_list += self.vanilla_ds.get_class_map(label)
            #################################################################

            self.subset_idx = list(set(examples_list))

        # Sort subset idx to make dataset deterministic (because set is unordered)
        self.subset_idx = sorted(self.subset_idx)

        # Generate self.class_map
        self.class_map = {}
        for c in range(1, 21):
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
        if self.base_stage:
            greater_pixel_idx = (target_tensor > max_val_label)
            novel_pixel_idx = torch.logical_and(target_tensor >= min_val_label,
                                                torch.logical_not(greater_pixel_idx))
            target_tensor[novel_pixel_idx] = 0
            target_tensor[greater_pixel_idx] -= len(self.val_label_set)
        else:
            lesser_pixel_idx = (target_tensor < min_val_label)
            greater_pixel_idx = (target_tensor > max_val_label)
            ignore_pixel_idx = (target_tensor == -1)
            target_tensor = target_tensor - (
                        min_val_label - 1)  # min_vis_label => 1 after this step
            target_tensor[lesser_pixel_idx] = 0
            target_tensor[greater_pixel_idx] = 0
            target_tensor[ignore_pixel_idx] = -1
        return target_tensor


def get_train_set(cfg):
    folding = cfg.DATASET.PASCAL5i.folding
    ds = VOC125iReader(utils.get_dataset_root(), folding, True, True, exclude_novel=True)
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    folding = cfg.DATASET.PASCAL5i.folding
    ds = VOC125iReader(utils.get_dataset_root(), folding, True, False, exclude_novel=False)
    return base_set(ds, "test", cfg)

def get_train_set_vanilla_label(cfg):
    folding = cfg.DATASET.PASCAL5i.folding
    ds = VOC125iReader(utils.get_dataset_root(), folding, True, True, exclude_novel=True, vanilla_label=True)
    return base_set(ds, "train", cfg)

def get_continual_train_set(cfg):
    ds = VOC12SegReader(utils.get_dataset_root(), True, download=True)
    return base_set(ds, "train", cfg)

def get_continual_test_set(cfg):
    ds = VOC12SegReader(utils.get_dataset_root(), False, download=True)
    return base_set(ds, "test", cfg)
