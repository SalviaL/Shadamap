import torch.nn as nn
import torch
        
class Model_SI(nn.Module):  # scael ignored
    def __init__(self, in_dim):
        super(Model_SI, self).__init__()
        self.h1 = nn.Sequential(nn.Linear(in_dim, 32), nn.Sigmoid())
        self.h2 = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(32, 1))

    def forward(self, x, mask=None, train=True):
        x = self.h1(x)
        x = self.h2(x)
        if train:
            return self.out(x)
        else:
            return x