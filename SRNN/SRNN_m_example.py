from SRNN_m import SRNN
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# data setting
data = pd.read_csv("PowerDemand.csv")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# window setting


class SlidingWindow(Dataset):
    def __init__(self, data, window):
        self.data = data
        self.window = window

    def __getitem__(self, index):
        x = self.data[index:index + self.window]
        return x

    def __len__(self):
        return len(self.data) - self.window


# DataLoader 생성
data = torch.tensor((np.array(data, dtype="float32")))
window_size = 50
batch_size = 32

dataset = SlidingWindow(data=data, window=window_size)
dataloader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=False)


# loss function
def srnn_loss(out_all, scaled_x_all) :
    loss = torch.zeros(1).cuda()
    for i in range(len(out_all)) :
        diff = out_all[i] - scaled_x_all[i]
        loss += (diff**2).mean()

    loss = loss/len(out_all)

    return loss


# model setting

n_input = 1
n_hidden = 8
n_layers = 3
nonlinearity = 'tanh'

model = SRNN(n_input, n_hidden, n_layers, nonlinearity=nonlinearity)

num_epochs = 1
learning_rate = 0.01

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

# training
k = []
for epoch in range(num_epochs):
    for i, window in enumerate(dataloader):
        window = window.to(device)
        out_all, scaled_x_all = model.forward(window)

        loss = srnn_loss(out_all, scaled_x_all)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        k.append(out_all)

        print("time window %d : loss = %f" %(i, loss.item()))