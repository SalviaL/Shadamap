import torch.nn as nn
import torch


class Weight(nn.Module):
    def __init__(self, mode) -> None:
        super(Weight, self).__init__()
        self.w1 = nn.Parameter(torch.tensor(0.5))
        self.w2 = nn.Parameter(torch.tensor(0.5))
        self.mode = mode

    def forward(self, l1, l2):
        mode = self.mode
        if mode == 'grid':
            return l1
        elif mode == 'unit':
            return l2
        elif mode == 'both':
            e_w1 = torch.exp(self.w1)
            e_w2 = torch.exp(self.w2)
            w1 = e_w1/(e_w1+e_w2)
            w2 = e_w2/(e_w1+e_w2)
            return w1*l1+w2*l2
        else:
            w = float(mode)
            return w*l1+(1-w)*l2

class IndWeight(nn.Module):
    def __init__(self, mode, n_unit) -> None:
        super(IndWeight,self).__init__()
        self.w11 = nn.Parameter(torch.tensor(0.5))
        self.w21 = nn.Parameter(torch.tensor(0.5))
        self.w1 = nn.Parameter(torch.rand(n_unit))
        self.w2 = nn.Parameter(torch.rand(n_unit))
        self.mode = mode
    def forward(self, l1, l2):
        mode = self.mode
        if mode == 'grid':
            return l1.mean()
        elif mode == 'unit':
            return l2.mean()
        elif mode == 'both':
            e_w1 = torch.exp(self.w1)
            e_w2 = torch.exp(self.w2)
            w1 = e_w1/(e_w1+e_w2)
            w2 = e_w2/(e_w1+e_w2)
            return (w1*l1+w2*l2).mean()
        elif mode == 'auto':
            e_w1 = torch.exp(self.w11)
            e_w2 = torch.exp(self.w21)
            w1 = e_w1/(e_w1+e_w2)
            w2 = e_w2/(e_w1+e_w2)
        else:
            w = float(mode)
            return (w*l1+(1-w)*l2).mean()
        
class Model_SI_old(nn.Module):  # scael ignored
    def __init__(self, in_dim):
        super(Model_SI_old, self).__init__()
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