from SRNN_s import SRNN
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import datetime

# data setting
data = pd.read_csv("PowerDemand.csv")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train, test = train_test_split(data, test_size=0.2, shuffle=False)

train_X = train[:-1]
train_y = train[1:]
test_X = test[:-1]
test_y = test[1:]

X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
train_X = X_scaler.fit_transform(train_X)
train_y = y_scaler.fit_transform(train_y)
test_X = X_scaler.fit_transform(test_X)


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
train_X = torch.tensor((np.array(train_X, dtype="float32")))
train_y = torch.tensor((np.array(train_y, dtype="float32")))
test_X = torch.tensor((np.array(test_X, dtype="float32")))
test_y = torch.tensor((np.array(test_y, dtype="float32")))

window_size = 50
batch_size = 32

train_dataset = SlidingWindow(X=train_X, y=train_y, window=window_size)
train_dataloader = DataLoader(dataset=train_dataset,
                        batch_size=batch_size,
                        shuffle=False)
test_dataset = SlidingWindow(X=test_X, y=test_y, window=window_size)
test_dataloader = DataLoader(dataset=test_dataset,
                        batch_size=batch_size,
                        shuffle=False)


# loss function
def mse_loss(out, y):

    out_sum = out.sum(dim=0)
    loss = ((out_sum - y) ** 2).mean()

    return loss


# model setting

n_input = 1
n_hidden = 1
n_layers = 3
nonlinearity = 'tanh'

model = SRNN(n_input, n_hidden, n_layers, nonlinearity=nonlinearity)

num_epochs = 1
learning_rate = 0.01

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training
for epoch in range(num_epochs):
    for i, data in enumerate(train_dataloader):
        window, y = data[0], data[1]
        window = window.to(device)
        y = y[:, -1, :].to(device)

        out = model.forward(window)

        loss = mse_loss(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # if i % 100 == 0 :
        #     print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        #           ". window : %d, loss : %1.5f" % (i, loss.item()))
        #     torch.save(model.state_dict(), "./gridsearch/model_state_dict.pt")

out_list, y_list = [], []
for i, data in enumerate(test_dataloader) :
    window, y = data[0], data[1]
    window = window.to(device)
    y = y[:,-1,:].to(device)

    out = model.forward(window)
    out = out.sum(dim=0)

    out_list.append(y_scaler.inverse_transform(out.cpu().detach().numpy()))
    y_list.append(y.cpu().numpy())

hat_y = np.concatenate(out_list)
real_y = np.concatenate(y_list)
hat_y = hat_y.flatten()
real_y = real_y.flatten()

((np.subtract(np.array([1,2,3]),np.array([1,1,1])))**2).mean()

# srnn_pred = pd.DataFrame({'real_y' : real_y, 'srnn_y' : hat_y})
# srnn_pred.to_csv('srnn_pred.csv')