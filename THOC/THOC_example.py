from THOC import THOC
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import datetime
from sklearn.preprocessing import MinMaxScaler

# data setting

data = torch.randn(3000, 9)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

# DataLoader 생성
data = torch.tensor((np.array(data, dtype = "float32")))
window_size = 50
batch_size = 32

dataset = SlidingWindow(data=data, window=window_size)
dataloader = DataLoader(dataset=dataset,
           batch_size=batch_size,
           shuffle=False)

# loss function

lambda_l2reg = 0.1
lambda_orth = 0.1
lambda_tss = 0.1

def thoc_loss(anomaly_scores, centroids_diff, out_of_drnn, timeseries_input) :

    # print("="*100)

    l2_loss = []
    for n, p in model.named_parameters() :
        if n == "cluster_centers" :
            for i in range(len(p)) :
                l2_loss.append(sum(torch.linalg.norm(p_[i], 2) for p_ in p))
        else :
            l2_loss.append(torch.linalg.norm(p, 2))
    loss_l2reg = torch.tensor(l2_loss).sum()

    loss_thoc = anomaly_scores.mean() + lambda_l2reg * loss_l2reg
    # print("loss_thoc : %f" % loss_thoc)

    loss_orth = centroids_diff.mean()
    # print("loss_orth : %f" % loss_orth)

    tss_list = []
    for i, out in enumerate(out_of_drnn) :
        dilation = 2**i
        for b in range(out_of_drnn.shape[2]) :
            tss = ((out[:-dilation, b] - timeseries_input[b, dilation:])**2).mean()
            tss_list.append(tss)
    loss_tss = torch.stack(tss_list).mean()
    # print("loss_tss : %f" % loss_tss)

    loss = loss_thoc + loss_orth * lambda_orth + loss_tss * lambda_tss
    # print("loss = %f" %loss)
    # print("=" * 100)
    return loss

# model setting

n_input = 9
n_hidden = 32
n_layers = 3
cell_type = 'RNN'
dropout = 0
n_centroids = [6, 4, 3]
tau = 1

init_data = data[:window_size*batch_size].view(batch_size, window_size, -1)
if torch.cuda.is_available() :
    init_data = init_data.to(device)

model = THOC(n_input, n_hidden, n_layers, n_centroids, init_data, tau=tau, dropout=dropout, cell_type=cell_type, batch_first=True)
if torch.cuda.is_available() :
    model = model.to(device)

# next(model.parameters()).is_cuda
# next(model.drnn.parameters()).is_cuda

num_epochs = 1
learning_rate = 0.001

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

'''
run this code if you want to check your model parameters

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)'''

# training
for epoch in range(num_epochs) :
    for i, window in enumerate(dataloader):
        if i == 0 :
            pass
        window = window.to(device)
        anomaly_scores, centroids_diff, out_of_drnn = model.forward(window)

        loss = thoc_loss(anomaly_scores, centroids_diff, out_of_drnn, window)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0 :
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ". Epochs : %d, windows : %d, loss : %1.5f" %(epoch, i, loss.item()))

