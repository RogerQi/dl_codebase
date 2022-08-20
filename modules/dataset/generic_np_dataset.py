import sys
import os
import numpy as np
import torch
from .baseset import base_set

class numpy_reader(object):
    '''
    Numpy reader that reads npy file.

    Reading Native NPY saved by np.save is much faster than pickle. ~ hdf5.
    '''
    def __init__(self, data_arr, label_arr):
        assert data_arr.shape[0] == label_arr.shape[0]
        self.data_arr = torch.from_numpy(data_arr).float()
        self.label_arr = torch.from_numpy(label_arr).float()

    def __getitem__(self, idx):
        return (self.data_arr[idx], self.label_arr[idx])

    def __len__(self):
        return self.data_arr.shape[0]


def get_train_set(cfg):
    data_npy_path = cfg.DATASET.NUMPY_READER.train_data_npy_path
    label_npy_path = cfg.DATASET.NUMPY_READER.train_label_npy_path
    if cfg.DATASET.NUMPY_READER.mmap:
        mmap_mode = "r"
    else:
        mmap_mode = None
    data_arr = np.load(data_npy_path, mmap_mode = mmap_mode, allow_pickle = False, fix_imports = False)
    label_arr = np.load(label_npy_path, mmap_mode = mmap_mode, allow_pickle = False, fix_imports = False)
    assert data_arr.shape[0] == label_arr.shape[0]
    ds = numpy_reader(data_arr, label_arr)
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    data_npy_path = cfg.DATASET.NUMPY_READER.test_data_npy_path
    label_npy_path = cfg.DATASET.NUMPY_READER.test_label_npy_path
    if cfg.DATASET.NUMPY_READER.mmap:
        mmap_mode = "r"
    else:
        mmap_mode = None
    data_arr = np.load(data_npy_path, mmap_mode = mmap_mode, allow_pickle = False, fix_imports = False)
    label_arr = np.load(label_npy_path, mmap_mode = mmap_mode, allow_pickle = False, fix_imports = False)
    assert data_arr.shape[0] == label_arr.shape[0]
    ds = numpy_reader(data_arr, label_arr)
    return base_set(ds, "test", cfg)