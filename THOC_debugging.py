# 1. Sliding Window Class 정상 작동하는가.
### torch rnn 모형에 Sliding window 적용해보고 parameter learning 되는지 확인.

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import torch

os.chdir("C:/Users/user/Desktop/model_git")

## data setting

data = pd.read_csv("Walk.csv")
data = data.iloc[:,2:-1]

## window setting

class SlidingWindow(Dataset) :
    def __init__(self, data, window) :
        self.data = data
        self.window = window
        
    def __getitem__(self, index) :
        x = self.data[index:index+self.window]
        return x
    
    def __len__(self) :
        return len(self.data) - self.window

## train_test_split (8:2)
k = int(len(data)*0.8)

train = data.iloc[:k,:]
test = data.iloc[k:,:]

# sliding window에 맞게 데이터 조정
window_size = 250

## DataLoader 생성
train_scaler, test_scaler = MinMaxScaler(), MinMaxScaler()
train = torch.tensor(train_scaler.fit_transform(np.array(train, dtype = "float32")))
test = torch.tensor(test_scaler.fit_transform(np.array(test, dtype="float32")))

train_dataset = SlidingWindow(data=train, window=window_size)
train_dl = DataLoader(dataset=train_dataset,
           batch_size=1,
           shuffle=False)

test_dataset = SlidingWindow(data=test, window=window_size)
test_dl = DataLoader(dataset=test_dataset,
           batch_size=1,
           shuffle=False)

## torch.rnn 모델 생성

input_size = 9
hidden_size = 9
num_layers = 3

rnn = nn.RNN(input_size= input_size, hidden_size = hidden_size, num_layers = num_layers)

num_epochs = 1
learning_rate = 0.01

optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

criterion = nn.MSELoss()

for epoch in range(num_epochs) :
    for i, window in enumerate(train_dl):
        window = window.type(torch.float32)
        window = Variable(window)
        optimizer.zero_grad()
        out, _ = rnn.forward(window)
        
        loss = criterion(window[0][1:], out[0][:-1])
        loss.backward()
        
        optimizer.step()
        
        for k in rnn.state_dict():
            print(k)
            print(f"{rnn.state_dict()[k]}")
            print("#"*100)
    
        if i ==1 :
            break
        
### Sliding Window Class는 정상 작동!

# 2. User defined loss function 정상 작동하는가.
### 위 모델에 User defined loss function 적용하여 parameter learning 확인.

## define temporal self-supervision loss

def tss_loss(output_of_rnn, input_data) :
    tss_loss = 0

    tss = output_of_rnn[:-1] - input_data[1:]
    tss = sum(sum(tss**2))
    tss_loss += tss/len(output_of_rnn[:-1])
    
    return tss_loss

## torch.rnn 모델 생성

input_size = 9
hidden_size = 9
num_layers = 3

rnn = nn.RNN(input_size= input_size, hidden_size = hidden_size, num_layers = num_layers)

num_epochs = 1
learning_rate = 0.01

optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

for epoch in range(num_epochs) :
    for i, window in enumerate(train_dl):
        window = window.type(torch.float32)
        window = Variable(window)
        optimizer.zero_grad()
        out, _ = rnn.forward(window)
        
        loss = tss_loss(out[0], window[0])
        loss.backward()
        
        optimizer.step()
        
        for k in rnn.state_dict():
            print(k)
            print(f"{rnn.state_dict()[k]}")
            print("#"*100)
    
        if i ==1 :
            break
        
# User defined loss function도 정상 작동..!