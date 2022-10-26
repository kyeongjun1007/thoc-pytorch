from THOC import THOC
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# data setting

data = torch.randn(100000, 9)


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

# sliding window에 맞게 데이터 조정
window_size = 100
batch_size = 1

# DataLoader 생성
data_scaler = MinMaxScaler()
data_tensor = torch.tensor(data_scaler.fit_transform(np.array(data, dtype="float32")))

dataset = SlidingWindow(data=data_tensor, window=window_size)
data_dl = DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=False)

lambda_orth = 0.1
lambda_tss = 0.1

# loss function
def thoc_loss(anomaly_scores, centroids_diff, out_of_drnn, timeseries_input):
    print("=" * 100)

    loss_thoc = anomaly_scores.mean()
    print("loss_thoc : %f" % loss_thoc)

    loss_orth = centroids_diff.mean()
    print("loss_orth : %f" % loss_orth)

    tss_list = []
    for i, out in enumerate(out_of_drnn):
        dilation = 2 ** i
        tss = ((out[:-dilation] - timeseries_input[0][dilation:]) ** 2).mean()
        tss_list.append(tss)
    loss_tss = torch.stack(tss_list).mean()
    print("loss_tss : %f" % loss_tss)

    loss = loss_thoc + loss_orth * lambda_orth + loss_tss * lambda_tss
    print("loss = %f" % loss)
    print("=" * 100)
    return loss


# model setting

n_input = 9
n_hidden = 9
n_layers = 3
cell_type = 'RNN'
dropout = 0
n_centroids = [6, 4, 3]
tau = 1

init_data = data[:window_size].view(batch_size, window_size, n_input)

model = THOC(n_input, n_hidden, n_layers, n_centroids, init_data, tau=tau, dropout=dropout,
             cell_type=cell_type, batch_first=True)

num_epochs = 30
learning_rate = 0.001

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

"""
run this code if you want to check your model parameters

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
"""

# training
for epoch in range(num_epochs):
    for i, window in enumerate(data_dl):
        if i == 0: # first window used to init cluster centroids
            pass

        anomaly_scores, centroids_diff, out_of_drnn = model.forward(window)

        loss = thoc_loss(anomaly_scores, centroids_diff, out_of_drnn, window)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("Epochs : %d, windows : %d  loss : %1.5f" % (epoch, i, loss.item()))

