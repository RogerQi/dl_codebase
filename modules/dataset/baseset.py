import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from .transforms.dispatcher import dispatcher

from IPython import embed

def binary_mask(target_tensor, fg_cls_idx):
    ignore_mask_idx = (target_tensor == -1)
    foreground_mask_idx = (target_tensor == fg_cls_idx)
    # Sanity check to make sure at least one foreground pixel is presented
    assert foreground_mask_idx.any()
    target_tensor = torch.zeros_like(target_tensor)
    target_tensor[foreground_mask_idx] = 1
    target_tensor[ignore_mask_idx] = -1
    return target_tensor

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
        self.dataset = dataset
        if split == "train":
            transforms_config_node = cfg.DATASET.TRANSFORM.TRAIN
        else:
            transforms_config_node = cfg.DATASET.TRANSFORM.TEST
        data_trans_ops, joint_trans_ops = dispatcher(transforms_config_node)
        self.data_transforms = self._get_mono_transforms(transforms_config_node, data_trans_ops)
        self.joint_transforms = self._get_joint_transforms(transforms_config_node, joint_trans_ops)
    
    def __getitem__(self, index):
        data, label = self.dataset[index]
        data = self.data_transforms(data)
        data, label = self.joint_transforms(data, label)
        return (data, label)
    
    def episodic_sample(self, n_shot):
        """
        Support 1-way few-shot segmentation only right now
        """
        assert hasattr(self.dataset, 'get_class_map')
        assert hasattr(self.dataset, 'visible_labels')
        assert hasattr(self.dataset, 'invisible_labels')
        sampled_class_id = random.choice(self.dataset.get_label_range())
        image_candidates = self.dataset.get_class_map(sampled_class_id)

        # random.sample samples without replacement and is faster than numpy.
        # 1 query image and n-shot support set.
        selected_images = random.sample(image_candidates, 1 + n_shot)
        query_img_chw, query_mask_hw = self[selected_images[0]]
        supp_img_mask_pairs_list = [self[i] for i in selected_images[1:]]
        supp_img_bchw, supp_mask_bhw = zip(*supp_img_mask_pairs_list)
        supp_img_bchw = torch.stack(supp_img_bchw)
        supp_mask_bhw = torch.stack(supp_mask_bhw)

        # Binary mask
        query_mask_hw = binary_mask(query_mask_hw, sampled_class_id)
        supp_mask_bhw = binary_mask(supp_mask_bhw, sampled_class_id)

        return {
            'sampled_class_id': sampled_class_id,
            'query_img_bchw': query_img_chw.view((1,) + query_img_chw.shape),
            'query_mask_bhw': query_mask_hw.view((1,) + query_mask_hw.shape),
            'supp_img_bchw': supp_img_bchw,
            'supp_mask_bhw': supp_mask_bhw
        }

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