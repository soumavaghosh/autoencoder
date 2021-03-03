from autoencoder_model import sparse_auto, auto, var_auto
import torch
from torch import optim
import numpy as np
import config
from read_data import read

data = read()

N = data.size()[0]

model = var_auto(data.shape[1], 'gaussian', config.dim)
model.train()
opt = optim.Adam(model.parameters(), lr = config.lr)

batch_size = config.batch_size
losses = []

epoch = config.epoch

def get_batch():
    i = 0
    while i<N:
        batch = data[i:min(i+batch_size, N), :]
        i+=len(batch)
        yield batch

for i in range(epoch):
    data = data[torch.randperm(data.size()[0])]
    
    for train_data in get_batch():
        opt.zero_grad()
    
        batch_indices = torch.LongTensor(np.random.randint(0, N, size=128))
        train_data = torch.index_select(data, 0, batch_indices)
    
        loss, kl , out , sample = model(train_data)
        loss = loss
        loss.backward()
        losses.append(loss.data.numpy())
    
        opt.step()
    print(f'Epoch - {i} loss value - {sum(losses)/len(losses)}')
    #print(str(i)+' - '+ str(loss.data)+' - '+str(kl.data))

torch.save(config.model_name)
