import pandas as pd
from autoencoder_model import sparse_auto, auto, var_auto
from sklearn.utils import shuffle
import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE

data = pd.read_csv('mnist_train.csv', header = 0)
data = np.array(data[list(data)[1:]])

data = torch.Tensor(data).type(torch.FloatTensor)/256.0
N = data.size()[0]

model = var_auto(data.shape[1], 'gaussian', 64)
model.train()
opt = optim.Adam(model.parameters(), lr = 0.005)

batch_size = 512
losses = []

epoch = 10000

for i in range(epoch):
    opt.zero_grad()

    batch_indices = torch.LongTensor(np.random.randint(0, N, size=128))
    train_data = torch.index_select(data, 0, batch_indices)

    loss, kl , out , sample = model(train_data)
    print(str(i)+' - '+ str(loss.data)+' - '+str(kl.data))
    loss = loss
    loss.backward()
    losses.append(loss.data.numpy())

    opt.step()

print('done')

def disp(i):
    sample = t_data[i]
    sample = torch.Tensor(sample.unsqueeze(0))
    loss, kl , out , sample= model(sample)
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