import sys
import numpy as np
import torch
import cv2

class base_set(torch.utils.data.Dataset):
    '''
    An implementation of torch.utils.data.Dataset that supports various
    data transforms and augmentation.
    '''
    def __init__(self, dataset, cfg):
        '''
        Args:
            dataset: any object with __getitem__ and __len__ methods implemented.
                Object retruned from dataset[i] is expected to be a raw torch tensor.
            cfg: yacs root config node object.
        '''
        self.dataset = dataset
        self.transforms = self._get_all_transforms(cfg)
    
    def __getitem__(self, index):
        desired_data = self.dataset[index]
        return self.apply_transforms(desired_data, self.transforms)

    def __len__(self):
        return len(self.dataset)

    def apply_transforms(self, img, transforms):
        '''
        Args:
            img: data to be transformed
            transforms: transforms return by self._get_all_transforms
        '''
        for trans in transforms:
            img = trans(img)
        return img

    def _get_all_transforms(self, cfg):
        return []