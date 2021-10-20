from torchvision import transforms

class dataset_normalization_wrapper(object):
    def __init__(self, cfg, ds):
        self.ds = ds
        self.to_tensor_func = transforms.ToTensor()
        self.normalizer = transforms.Normalize(mean=cfg.DATASET.TRANSFORM.TRAIN.TRANSFORMS_DETAILS.NORMALIZE.mean,
                                    std=cfg.DATASET.TRANSFORM.TRAIN.TRANSFORMS_DETAILS.NORMALIZE.sd)

    def __getitem__(self, idx):
        data, label = self.ds[idx]
        data = self.to_tensor_func(data)
        data = self.normalizer(data)
        return data, label

    def __len__(self):
        return len(self.ds)
