import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class auto(nn.Module):

    def __init__(self, inp_size, hidden_size, rho=0.05):
        super(auto, self).__init__()

        self.inp_size = inp_size
        self.hidden_size = hidden_size
        self.enc = nn.Linear(self.inp_size, self.hidden_size, bias = True)
        self.dec = nn.Linear(self.hidden_size, self.inp_size, bias=True)

        self.loss = nn.MSELoss()
        self.rho = rho

    def forward(self, input):
        h = self.enc(input)
        h = F.sigmoid(h)
        z = self.dec(h)

        loss_val = self.calc_loss(h, z)

        return loss_val, h, z

    def calc_loss(self, z, input):

        loss_val = self.loss(z, input)
        loss_val = torch.mean(loss_val)

        return loss_val

class sparse_auto(nn.Module):

    def __init__(self, inp_size, hidden_size=64, rho=0.05):
        super(sparse_auto, self).__init__()

        self.inp_size = inp_size
        self.hidden_size = hidden_size
        self.enc = nn.Linear(self.inp_size, self.hidden_size, bias = True)
        self.dec = nn.Linear(self.hidden_size, self.inp_size, bias=True)

        self.loss = nn.MSELoss()
        self.rho = rho

    def forward(self, input):
        h = self.enc(input)
        h = F.sigmoid(h)
        z = self.dec(h)

        loss_val = self.calc_loss(h, z, input)

        return loss_val, h, z

    def calc_loss(self, h, z, input):

        loss_val = self.loss(z, input)
        h = torch.mean(h, 0)
        kl = h * torch.log(h/self.rho) + (1-h) * torch.log((1-h)/(1-self.rho))
        kl = torch.mean(kl)
        loss_val = torch.mean(loss_val)

        return loss_val + kl


