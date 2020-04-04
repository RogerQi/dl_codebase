import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class net(nn.Module):
    def __init__(self, cfg):
        # Similar Structure from
        # https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf
        super(net, self).__init__()
        self.aux_dict = {}
        input_size = np.prod(cfg.input_dim)
        self.linear1 = nn.Linear(input_size, 500)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(500, 120)
        self.relu2 = nn.ReLU()
        self.linear3_mu = nn.Linear(120, 30)
        self.linear3_log_var = nn.Linear(120, 30)
        # sampling happens here
        self.standard_normal = torch.distributions.normal.Normal(0, 1)
        self.inv_linear3 = nn.Linear(30, 120)
        self.inv_relu2 = nn.ReLU()
        self.inv_linear2 = nn.Linear(120, 500)
        self.inv_relu1 = nn.ReLU()
        self.inv_linear1 = nn.Linear(500, input_size)
    
    def encode(self, x):
        x = torch.flatten(x, start_dim = 1)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        return [self.linear3_mu(x), self.linear3_log_var(x)] # mu, var
    
    def sample(self, mean_vec, log_var_vec):
        # Sampling with Reparameterization Trick
        assert mean_vec.shape == log_var_vec.shape
        offset_vec = self.standard_normal.sample(mean_vec.shape).to(mean_vec.device)
        x = mean_vec + offset_vec * torch.exp(log_var_vec * 0.5)
        return x
    
    def decode(self, x):
        # inverse
        x = self.inv_linear3(x)
        x = self.inv_relu2(x)
        x = self.inv_linear2(x)
        x = self.inv_relu1(x)
        x = self.inv_linear1(x)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        self.aux_dict['mean_vec'] = mu # for training purpose
        self.aux_dict['log_var_vec'] = log_var
        x = self.sample(mu, log_var)
        x = self.decode(x)
        return x