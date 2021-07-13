import torch

from abc import abstractmethod

class trainer_base:
    def __init__(self, cfg, backbone_net, post_processor, criterion, device):
        pass

    @abstractmethod
    def train_one(self, device, train_loader, optimizer, epoch):
        pass

    @abstractmethod
    def val_one(self, device, val_loader):
        pass

    @abstractmethod
    def test_one(self, device, test_loader):
        pass

    def save_model(self, file_path):
        torch.save({
            "backbone": self.backbone_net.state_dict(),
            "head": self.post_processor.state_dict()
        }, file_path)  
