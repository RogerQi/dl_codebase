import os
import random
import numpy as np
import torch

from .baseset import base_set

def binary_mask(target_tensor, fg_cls_idx):
    ignore_mask_idx = (target_tensor == -1)
    foreground_mask_idx = (target_tensor == fg_cls_idx)
    # Sanity check to make sure at least one foreground pixel is presented
    assert foreground_mask_idx.any()
    target_tensor = torch.zeros_like(target_tensor)
    target_tensor[foreground_mask_idx] = 1
    target_tensor[ignore_mask_idx] = -1
    return target_tensor

def fs_seg_episodic_sample(vanilla_ds, n_way, n_shot, n_query):
    """
    Support 1-way few-shot segmentation only right now
    """
    assert n_way == 1
    assert n_query == 1
    assert hasattr(vanilla_ds.dataset, 'get_label_range')
    assert hasattr(vanilla_ds.dataset, 'get_class_map')
    assert isinstance(vanilla_ds, base_set) # assert it's a wrapper so vanilla_ds.dataset can be accessed
    sampled_class_id = random.choice(vanilla_ds.dataset.get_label_range())
    image_candidates = vanilla_ds.dataset.get_class_map(sampled_class_id)

    # random.sample samples without replacement and is faster than numpy.
    # 1 query image and n-shot support set.
    selected_images = random.sample(image_candidates, 1 + n_shot)
    query_img_chw, query_mask_hw = vanilla_ds[selected_images[0]]
    supp_img_mask_pairs_list = [vanilla_ds[i] for i in selected_images[1:]]
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

def fs_classification_episodic_sample(vanilla_ds, n_way, n_shot, n_query):
    assert isinstance(vanilla_ds, base_set) # assert it's a wrapper so vanilla_ds.dataset can be accessed
    assert hasattr(vanilla_ds.dataset, 'get_label_range')
    assert hasattr(vanilla_ds.dataset, 'get_class_map')

    label_candidates = vanilla_ds.dataset.get_label_range()
    sampled_labels = np.random.choice(label_candidates, size = (n_way, ), replace=False)

    supp_set = []
    query_set = []

    for label_idx, label in enumerate(sampled_labels):
        image_candidates = vanilla_ds.dataset.get_class_map(label)
        selected_samples = random.sample(image_candidates, n_query + n_shot)
        img_label_list = [list(vanilla_ds[i]) for i in selected_samples]
        for i in range(len(img_label_list)):
            img_label_list[i][1] = label_idx
        supp_set += img_label_list[:n_shot]
        query_set += img_label_list[n_shot:]
    
    supp_img_bchw, supp_label_b = zip(*supp_set)
    query_img_bchw, query_label_b = zip(*query_set)

    return {
        'supp_img_bchw': torch.stack(supp_img_bchw),
        'supp_label_b': torch.LongTensor(supp_label_b),
        'query_img_bchw': torch.stack(query_img_bchw),
        'query_label_b': torch.LongTensor(query_label_b)
    }

def fs_clf_fg_bg_episodic_sample(fg_ds, bg_ds, n_shot, n_query):
    assert isinstance(fg_ds, base_set) # assert it's a wrapper so vanilla_ds.dataset can be accessed
    assert isinstance(bg_ds, base_set) # assert it's a wrapper so vanilla_ds.dataset can be accessed
    assert hasattr(fg_ds.dataset, 'get_label_range')
    assert hasattr(fg_ds.dataset, 'get_class_map')

    label_candidates = fg_ds.dataset.get_label_range()
    sampled_label = np.random.choice(label_candidates)

    fg_img_candidates = fg_ds.dataset.get_class_map(sampled_label)
    selected_samples = random.sample(fg_img_candidates, n_query + n_shot)
    fg_img_label_list = [list(fg_ds[i]) for i in selected_samples]
    for i in range(len(fg_img_label_list)):
        fg_img_label_list[i][1] = 1 # fg are marked as 1
    
    selected_bg_samples = np.random.choice(len(bg_ds), size = (n_query + n_shot,), replace=False)
    bg_img_label_list = [list(bg_ds[i]) for i in selected_bg_samples]
    for i in range(len(bg_img_label_list)):
        bg_img_label_list[i][1] = 0
    
    supp_img_label_list = bg_img_label_list[:n_shot] + fg_img_label_list[:n_shot]
    query_img_label_list = bg_img_label_list[n_shot:] + fg_img_label_list[n_shot:]

    supp_img_bchw, supp_label_b = zip(*supp_img_label_list)
    query_img_bchw, query_label_b = zip(*query_img_label_list)

    return {
        'supp_img_bchw': torch.stack(supp_img_bchw),
        'supp_label_b': torch.LongTensor(supp_label_b),
        'query_img_bchw': torch.stack(query_img_bchw),
        'query_label_b': torch.LongTensor(query_label_b)
    }

class loader_wrapper(object):
    def __init__(self, vanilla_ds, n_iter, n_way, n_shot, n_query, sampler_func):
        self.vanilla_ds = vanilla_ds
        self.n_iter = n_iter
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.sampler_func = sampler_func
    
    def __len__(self):
        return self.n_iter

    def __getitem__(self, idx):
        return self.sampler_func(self.vanilla_ds, self.n_way, self.n_shot, self.n_query)

class fg_bg_loader_wrapper(object):
    def __init__(self, fg_ds, bg_ds, n_iter, n_shot, n_query, sampler_func):
        self.fg_ds = fg_ds
        self.bg_ds = bg_ds
        self.n_iter = n_iter
        self.n_shot = n_shot
        self.n_query = n_query
        self.sampler_func = sampler_func
    
    def __len__(self):
        return self.n_iter

    def __getitem__(self, idx):
        return self.sampler_func(self.fg_ds, self.bg_ds, self.n_shot, self.n_query)

def get_fs_seg_loader(vanilla_ds, n_iter, n_way, n_shot, n_query):
    ds = loader_wrapper(vanilla_ds, n_iter, n_way, n_shot, n_query, fs_seg_episodic_sample)
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x : x)

def get_continual_seg_loader(vanilla_ds, n_iter, n_way, n_shot, n_query):

    raise NotImplementedError

def get_fs_classification_loader(vanilla_ds, n_iter, n_way, n_shot, n_query):
    # 5 way
    # n_shot: 5 per class
    # n_query: 15 per class
    ds = loader_wrapper(vanilla_ds, n_iter, n_way, n_shot, n_query, fs_classification_episodic_sample)
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x : x)

def get_fs_clf_fg_bg_loader(vanilla_ds, bg_ds, n_iter, n_shot, n_query):
    ds = fg_bg_loader_wrapper(vanilla_ds, bg_ds, n_iter, n_shot, n_query, fs_clf_fg_bg_episodic_sample)
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x : x)
