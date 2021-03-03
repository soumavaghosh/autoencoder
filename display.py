#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 01:31:12 2021

@author: soumavaghosh
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import config

model = torch.save(config.model_name)

def disp(i):
    sample = t_data[i]
    sample = torch.Tensor(sample.unsqueeze(0))
    loss, kl , out , sample = model(sample)
    out = out*256
    out = out.type(torch.LongTensor).clamp(0,256)
    out = out.data.numpy().reshape(28,28)
    plt.imshow(out)

test = pd.read_csv('mnist_train.csv', header = 0)
t_data = np.array(test[list(test)[1:]])

t_data = torch.Tensor(t_data).type(torch.FloatTensor)/256.0
loss, kl , out, sample = model(t_data)

sample = sample.data.numpy()
sample_tsne = TSNE(n_components=2, verbose = 5).fit_transform(sample)

i = 1500
print(test.iloc[i]['label'])
plt.imshow(t_data[i].reshape(28, 28))
disp(i)

n = 10
colors = cm.rainbow(np.linspace(0, 1, n))
fig, ax = plt.subplots()

for i in range(n):
#    t_data = np.array(test[test['label']==i][list(test)[1:]])
#    t_data = torch.Tensor(t_data).type(torch.FloatTensor)/256.0
#    loss, kl , out, sample = model(t_data)
    ind = np.array(test[test['label']==i].index)
    sample1 = sample_tsne[ind,:]
    
    ax.scatter(sample1[:,0], sample1[:,1], color=colors[i], label = i)
    
ax.legend()
plt.show()