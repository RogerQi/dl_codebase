import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class net(nn.Module):
    def __init__(self, cfg):
        # Similar Structure from
        # https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf
        super(net, self).__init__()
        input_size = np.prod(cfg.input_dim)
        self.linear1 = nn.Linear(input_size, 500)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(500, 120)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(120, 60)
        self.inv_linear3 = nn.Linear(60, 120)
        self.inv_relu2 = nn.ReLU()
        self.inv_linear2 = nn.Linear(120, 500)
        self.inv_relu1 = nn.ReLU()
        self.inv_linear1 = nn.Linear(500, input_size)


    def forward(self, x):
        x = torch.flatten(x, start_dim = 1)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.inv_linear3(x)
        x = self.inv_relu2(x)
        x = self.inv_linear2(x)
        x = self.inv_relu1(x)
        x = self.inv_linear1(x)
        return x