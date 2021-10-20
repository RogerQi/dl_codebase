import torch

from abc import abstractmethod

class trainer_base:
    def __init__(self, cfg, backbone_net, post_processor, criterion, dataset_module, device):
        self.cfg = cfg
        self.backbone_net = backbone_net
        self.post_processor = post_processor
        self.criterion = criterion
        self.device = device
        self.feature_shape = self.backbone_net.get_feature_tensor_shape(self.device)

        # Obtain dataset
        self.train_set = dataset_module.get_train_set(cfg)
        self.val_set = dataset_module.get_val_set(cfg)

        print("Training set contains {} data points. Val set contains {} data points.".format(len(self.train_set), len(self.val_set)))

        # Prepare loaders
        self.use_cuda = not cfg.SYSTEM.use_cpu
        self.loader_kwargs = {'num_workers': cfg.SYSTEM.num_workers, 'pin_memory': cfg.SYSTEM.pin_memory} if self.use_cuda else {}

        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=cfg.TRAIN.batch_size, shuffle=True, **self.loader_kwargs)
        self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=cfg.TEST.batch_size, shuffle=False, **self.loader_kwargs)

    @abstractmethod
    def train_one(self, device, optimizer, epoch):
        raise NotImplementedError

    @abstractmethod
    def val_one(self, device):
        raise NotImplementedError

    @abstractmethod
    def test_one(self, device):
        raise NotImplementedError

    @abstractmethod
    def live_run(self, device):
        raise NotImplementedError

    def save_model(self, file_path):
        """Save default model (backbone_net, post_processor to a specified file path)

        Args:
            file_path (str): path to save the model
        """
        torch.save({
            "backbone": self.backbone_net.state_dict(),
            "head": self.post_processor.state_dict()
        }, file_path)
    
    def load_model(self, file_path):
        """Load weights for default model components (backbone_net, post_process) from a given file path

        Args:
            file_path (str): path to trained weights
        """
        trained_weight_dict = torch.load(file_path, map_location=self.device)
        self.backbone_net.load_state_dict(trained_weight_dict['backbone'], strict=True)
        self.post_processor.load_state_dict(trained_weight_dict['head'], strict=True)
