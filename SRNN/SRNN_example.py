from SRNN import SRNN
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# data setting
data = torch.randn(100, 9)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

# model setting

n_input = 9
n_hidden = 32
n_layers = 3
nonlinearity = 'tanh'

model = SRNN(n_input, n_hidden, n_layers, nonlinearity=nonlinearity)

num_epochs = 1
learning_rate = 0.001

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training
for epoch in range(num_epochs):
    for i, window in enumerate(dataloader):
        window = window.to(device)
        out = model.forward(window)

        print(out)
