import torch
import torch.nn as nn
import torch.nn.functional as F

class fc(nn.Module):
    def __init__(self, cfg, feature_size):
        super(fc, self).__init__()
        output_size = cfg.num_classes
        latent_space_dim = list(cfg.CLASSIFIER.FC.hidden_layers)
        latent_space_dim = [feature_size] + latent_space_dim
        latent_space_dim = latent_space_dim + [output_size]
        net_list = []
        for i in range(len(latent_space_dim) - 1):
            net_list.append(nn.Linear(latent_space_dim[i], latent_space_dim[i + 1], bias = cfg.CLASSIFIER.FC.bias))
            if i != len(latent_space_dim) - 2:
                net_list.append(nn.ReLU())
        print(net_list)
        self.net = nn.Sequential(*net_list)

    def forward(self, x):
        x = torch.flatten(x, start_dim = 1)
        return self.net(x)