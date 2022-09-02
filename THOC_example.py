import os
os.chdir("C:/Users/user/Desktop/model_git")

from THOC import THOC
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

# data setting

data = pd.read_csv("Walk.csv")
data = data.iloc[:,2:-1]

# window setting

class SlidingWindow(Dataset) :
    def __init__(self, data, window) :
        self.data = data
        self.window = window
        
    def __getitem__(self, index) :
        x = self.data[index:index+self.window]
        return x
    
    def __len__(self) :
        return len(self.data) - self.window

# train_test_split (8:2)
k = int(len(data)*0.8)

train = data.iloc[:k,:]
test = data.iloc[k:,:]

# sliding window에 맞게 데이터 조정
window_size = 250

# DataLoader 생성
train = torch.tensor(np.array(train, dtype=float))
test = torch.tensor(np.array(test, dtype=float))

train_dataset = SlidingWindow(data=train, window=window_size)
train_dl = DataLoader(dataset=train_dataset,
           batch_size=1,
           shuffle=False)

test_dataset = SlidingWindow(data=test, window=window_size)
test_dl = DataLoader(dataset=test_dataset,
           batch_size=1,
           shuffle=False)

# loss function
def thoc_loss(anomaly_scores, cluster_centroids, out_of_drnn, timeseries_input) :
    
    loss_thoc = sum(anomaly_scores)/(window_size*n_centroids[-1])
    
    loss_orth = 0
    for l in range(len(cluster_centroids)) :
        C = cluster_centroids[l]
        CI = torch.matmul(torch.t(C),C) - torch.eye(C.size(1))
        CI = torch.norm(CI, p='fro')
        loss_orth += (CI*CI)
    loss_orth = loss_orth/len(cluster_centroids)
    
    loss_tss = 0
    for i, out in enumerate(out_of_drnn) :
        dilation = 2**i
        tss = out[:-dilation] - timeseries_input[0][dilation:]
        tss = sum(sum(tss**2))
        loss_tss += tss/len(out[:-dilation])
    loss_tss/len(out_of_drnn)
    
    loss = loss_thoc + loss_orth + loss_tss
    
    return loss
    

#sum(sum((out_of_drnn[0][:-245] - train_dataset[0][245:])**2))
#torch.tensor([1,2,3,4,5])[1:]

# model setting

n_input = 9
n_hidden = 9
n_layers = 3
cell_type = 'RNN'
dropout = 0
n_centroids = [6,4,3]

model = THOC(n_input, n_hidden, n_layers, n_centroids, dropout = dropout, cell_type = cell_type)

num_epochs = 100
learning_rate = 0.001

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training
for epoch in range(num_epochs) :
    for i, window in enumerate(train_dl):
        window = window.type(torch.float32)
        window = Variable(window)
        optimizer.zero_grad()
        if i == 0 :
            anomaly_scores, cluster_centroids, out_of_drnn = model.forward(window, first = True)
        else : 
            anomaly_scores, cluster_centroids, out_of_drnn = model.forward(window)
        loss = Variable(thoc_loss(anomaly_scores, cluster_centroids, out_of_drnn, window), requires_grad=True)
        loss.backward()
    
        optimizer.step()
        if i % 30 == 0 :
            print("window steps : %d, loss : %1.5f" %(i, loss.item()))
    print("Epochs : %d, loss : %1.5f" %(epoch, loss.item()))
    