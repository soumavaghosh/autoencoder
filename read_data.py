#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 02:20:25 2021

@author: soumavaghosh
"""
import pandas as pd
import numpy as np
import torch

def read():

    data = pd.read_csv('mnist_train.csv', header = 0)
    data = np.array(data[list(data)[1:]])
    
    data = torch.Tensor(data).type(torch.FloatTensor)/256.0
    
    return data