import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from .transforms.dispatcher import dispatcher

class base_set(torch.utils.data.Dataset):
    '''
    An implementation of torch.utils.data.Dataset that supports various
    data transforms and augmentation.
    '''
    def __init__(self, dataset, split, cfg):
        '''
        Args:
            dataset: any object with __getitem__ and __len__ methods implemented.
                Object retruned from dataset[i] is expected to be (raw_tensor, label).
            split: ("train" or "test"). Specify dataset mode
            cfg: yacs root config node object.
        '''
        assert split in ["train", "test"]
        self.cfg = cfg
        self.data_cache_flag = cfg.DATASET.cache_all_data
        self.dataset = dataset
        if split == "train":
            transforms_config_node = cfg.DATASET.TRANSFORM.TRAIN
        else:
            transforms_config_node = cfg.DATASET.TRANSFORM.TEST
        data_trans_ops, joint_trans_ops = dispatcher(transforms_config_node)
        self.data_transforms = self._get_mono_transforms(transforms_config_node, data_trans_ops)
        self.joint_transforms = self._get_joint_transforms(transforms_config_node, joint_trans_ops)
        if self.data_cache_flag:
            self.cached_dataset = {}
    
    def __getitem__(self, index):
        if self.data_cache_flag:
            if index in self.cached_dataset:
                data, label = self.cached_dataset[index]
            else:
                data, label = self.dataset[index]
                self.cached_dataset[index] = (data, label) # Write to memory
        else:
            data, label = self.dataset[index]
        data = self.data_transforms(data)
        data, label = self.joint_transforms(data, label)
        return (data, label)

    def __len__(self):
        return len(self.dataset)

    def _get_mono_transforms(self, transforms_cfg, transform_ops_list):
        transforms_list = transforms_cfg.transforms
        assert len(transforms_list) != 0
        if transforms_list == ('none',):
            return transforms.Compose([])
        if transforms_list == ('normalize'):
            return transforms.Compose([self._get_dataset_normalizer(transforms_cfg)])
        # Nontrivial transforms...
        try:
            normalize_first_occurence = transforms_list.index("normalize")
            assert normalize_first_occurence == len(transforms_list) - 1, "normalization happens last"
            return transforms.Compose([transforms.ToPILImage()] + transform_ops_list + [transforms.ToTensor(),
                        self._get_dataset_normalizer(transforms_cfg)])
        except ValueError:
            # Given transforms does not contain normalization
            return transforms.Compose([transforms.ToPILImage()] + transform_ops_list + [transforms.ToTensor()])
    
    def _get_joint_transforms(self, transforms_cfg, transforms_ops_list):
        if len(transforms_ops_list) == 0:
            def composed_func(img, target):
                return img, target
        else:
            def composed_func(img, target):
                for func in transforms_ops_list:
                    img, target = func(img, target)
                return img, target
            return composed_func
        return composed_func

    def _get_dataset_normalizer(self, transforms_cfg):
        return transforms.Normalize(transforms_cfg.TRANSFORMS_DETAILS.NORMALIZE.mean,
                                    transforms_cfg.TRANSFORMS_DETAILS.NORMALIZE.sd)