from SRNN_singlepred import SRNN
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import datetime

# data setting
data = pd.read_csv("PowerDemand.csv")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

X = data[:-1]
y = data[1:]

scaler = MinMaxScaler()
X = scaler.fit_transform(X)


# window setting

class SlidingWindow(Dataset):
    def __init__(self, X, y, window):
        self.X = X
        self.y = y
        self.window = window

    def __getitem__(self, index):
        x_out = self.X[index:index + self.window]
        y_out = self.y[index:index + self.window]
        return (x_out, y_out)

    def __len__(self):
        return len(self.X) - self.window


# DataLoader 생성
X = torch.tensor((np.array(X, dtype="float32")))
y = torch.tensor((np.array(y, dtype="float32")))

window_size = 50
batch_size = 32

dataset = SlidingWindow(X=X, y=y, window=window_size)
dataloader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=False)


# loss function
def srnn_loss(out, y):
    out_sum = out.sum(dim=0)
    loss = ((out_sum - y) ** 2).mean()

    return loss


# model setting

n_input = 1
n_hidden = 128
n_layers = 3
nonlinearity = 'tanh'

model = SRNN(n_input, n_hidden, n_layers, nonlinearity=nonlinearity)

num_epochs = 10
learning_rate = 0.01

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training
k = []
for epoch in range(num_epochs):
    if epoch == 0 :
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    for i, data in enumerate(dataloader):
        window, y = data[0], data[1]
        window = window.to(device)
        y = y[:, -1, :].to(device)

        out = model.forward(window)

        loss = srnn_loss(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
          ". Epochs : %d, loss : %1.5f" % (epoch, loss.item()))
