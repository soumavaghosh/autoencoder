import pandas as pd
from autoencoder_model import sparse_auto, auto
from sklearn.utils import shuffle
import torch
from torch import optim

data = pd.read_csv()

model = sparse_auto(data.shape[0])
model.train()
opt = optim.Adam(model.parameters(), lr = 0.005)

batch_size = 64
losses = []

epoch = 100000

for _ in range(epoch):
    opt.zero_grad()

    data = shuffle(data)
    train_data = data[:batch_size]

    loss, _, _ = model(train_data)
    loss.backward()
    losses.append(loss)

    opt.step()
    print(loss)