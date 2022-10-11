# 0. Benchmark RNN model  
### torch rnn 모형 작성
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

## train_test_split (8:2)
k = int(len(data)*0.8)

train = data.iloc[:k,:]
test = data.iloc[k:,:]

check = data.iloc[:100,:]
check = torch.tensor(np.array(check), dtype = torch.float32)
check = torch.stack([check])

train_data = torch.tensor(np.array(train), dtype=torch.float32)
train_data = torch.stack([train_data])
test_data = torch.tensor(np.array(test), dtype=torch.float32)
test_data = torch.stack([test_data])

## torch.rnn 모델 생성

input_size = 9
hidden_size = 9
num_layers = 3

rnn = nn.RNN(input_size= input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True)

num_epochs = 5
learning_rate = 0.01

optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

criterion = nn.MSELoss()

hidden = None

target = test_data[0][1:]

#torch.autograd.set_detect_anomaly(False)
for epoch in range(num_epochs) :        
    out, hidden = rnn.forward(test_data, hidden)
    loss = criterion(target, out[0][:-1])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(loss.item())
        
# 1. Sliding Window Class 정상 작동하는가.
### torch rnn 모형에 Sliding window 적용해보고 parameter learning 되는지 확인.

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

rnn = nn.RNN(input_size= input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True)

num_epochs = 1
learning_rate = 0.01

optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

criterion = nn.MSELoss()

hidden = None
for epoch in range(num_epochs) :
    for i, window in enumerate(train_dl):
        out, hidden = rnn.forward(window ,hidden)
        loss = criterion(window[0][1:], out[0][:-1])
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(loss.item())

        
### 한 epoch에서 window가 slide하면서 여러번 학습하는 과정에서 에러 발생 (연산그래프를 두번이상 그리는게 에러로 잡힘)

# 2. User defined loss function 정상 작동하는가.
### rnn 모델에 User defined loss function 적용하여 parameter learning 확인. (window X)

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

hidden = None
for epoch in range(num_epochs) :
    out, hidden = rnn.forward(train_data, hidden)
    loss = criterion(train_data[0][1:], out[0][:-1])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(loss.item())
        
### User defined loss function은 정상 작동..!

# 3. Dilated RNN without Sliding Window & User defined loss function

n_input = 9
n_hidden = 9
n_layers = 3
cell_type = 'RNN'

model = DRNN(n_input, n_hidden, n_layers, cell_type=cell_type, batch_first = True)

num_epochs = 1
learning_rate = 0.01

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

criterion = nn.MSELoss()

# training
hidden = None
for epoch in range(num_epochs) :
    for i, window in enumerate(train_dl):
        # window = window.type(torch.float32)
        # window = Variable(window)
        
        out, hidden = model.forward(window, hidden)
        loss = criterion(window[0][1:], out[0][:-1])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        for k in model.state_dict():
            print(k)
            print(f"{model.state_dict()[k]}")
            print("#"*100)
    
        if i ==1 :
            break

### hidden-hidden weight만 학습이 안되네요..

print("\n"*10000)
