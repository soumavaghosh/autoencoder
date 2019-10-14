import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal

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

class var_auto(nn.Module):

    def __init__(self, inp_size, dist='gaussian', hidden_size=64):
        super(var_auto, self).__init__()

        self.inp_size = inp_size
        self.hidden_size = hidden_size
        self.enc1 = nn.Linear(self.inp_size, 256, bias = True)
        self.enc2 = nn.Linear(256, 128, bias = True)
        self.mean_net = nn.Linear(128, self.hidden_size, bias=True)
        self.std_net = nn.Linear(128, self.hidden_size, bias=True)
        self.dec1 = nn.Linear(self.hidden_size, 128, bias=True)
        self.dec2 = nn.Linear(128, 256, bias=True)
        self.dec3 = nn.Linear(256, self.inp_size, bias=True)
        self.dist = dist
        self.loss1 = nn.MSELoss()
        self.loss2 = nn.BCELoss()

        self.init_emb()

    def init_emb(self):

        initrange = 0.01
        self.enc1.weight.data.uniform_(-initrange, initrange)
        self.enc2.weight.data.uniform_(-initrange, initrange)
        self.mean_net.weight.data.uniform_(-initrange, initrange)
        self.std_net.weight.data.uniform_(-initrange, initrange)
        self.dec1.weight.data.uniform_(-initrange, initrange)
        self.dec2.weight.data.uniform_(-initrange, initrange)
        self.dec3.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        h = self.enc1(input)
        h = F.relu(h)
        h = self.enc2(h)
        h = F.relu(h)
        mean = self.mean_net(h)
        std = F.softplus(self.std_net(h))+0.0000001

        sample = Normal(0, 1).sample((input.shape[0], self.hidden_size))
        sample = sample * std + mean
        out = self.dec1(sample)
        out = F.relu(out)
        out = self.dec2(out)
        out = F.relu(out)
        out = self.dec3(out)
        if self.dist == 'bernoulli':
            out = F.sigmoid(out)
        else:
            out = F.relu(out)
        loss_v, kl = self.calc_loss(input, std, mean, out)

        return loss_v, kl, out, sample

    def calc_loss(self, input, std, mean, out):
        if self.dist == 'gaussian':
            loss_val = torch.mean(self.loss1(out, input))
        elif self.dist == 'bernoulli':
            loss_val = torch.mean(self.loss2(out, input))
        kl = torch.mean(torch.pow(mean, 2)/2 + torch.pow(std, 2)/2 - torch.log(std) -0.5)

        return loss_val, kl
