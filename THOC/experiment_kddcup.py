import pandas as pd
from THOC import THOC
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# data setting
pd_df = pd.read_csv("./kddcup.csv", header=None, index_col=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pd_df = pd_df[pd_df.columns.difference([1,2,3,6,11,20,21])]

pd_df.loc[pd_df[41] != 'normal.'] = 0
pd_df.loc[pd_df[41] == 'normal.'] = 1

X = pd_df.iloc[:,:-1]
y = pd_df.iloc[:,-1]

X_train, X, y_train, y = train_test_split(X, y, test_size=0.5, shuffle=False)
X_valid, X_test, y_valid, y_test = train_test_split(X, y, test_size=0.5, shuffle=False)

del(pd_df, X, y)

# window setting

class SlidingWindow(Dataset):
    def __init__(self, X, y, window):
        self.X = X
        self.y = y
        self.window = window

    def __getitem__(self, index):
        out_X = self.X[index:index + self.window]
        out_y = self.y[index:index + self.window]
        return (out_X, out_y)

    def __len__(self):
        return len(self.X) - self.window


# DataLoader 생성
X_train_tensor = torch.tensor((np.array(X_train, dtype="float32")))
X_valid_tensor = torch.tensor((np.array(X_valid, dtype="float32")))
y_train_tensor = torch.tensor((np.array(y_train, dtype="float32")))
y_valid_tensor = torch.tensor((np.array(y_valid, dtype="float32")))

window_size = 80
batch_size = 32 ##

train_dataset = SlidingWindow(X=X_train_tensor, y=y_train_tensor, window=window_size)
valid_dataset = SlidingWindow(X=X_valid_tensor, y=y_valid_tensor, window=window_size)

train_dl = DataLoader(dataset=train_dataset,
                        batch_size=batch_size,
                        shuffle=False)
valid_dl = DataLoader(dataset=valid_dataset,
                        batch_size=batch_size,
                        shuffle=False)

# loss function

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

n_input = 34
n_hidden = 32 ##
n_layers = 3
cell_type = 'RNN'
dropout = 0
n_centroids = [6,4,3] ##
tau = 1 ##

lambda_l2reg = 1e-06
lambda_orth = 0.01 ##
lambda_tss = 0.01 ##
num_epochs = 5
learning_rate = 0.001 ##

init_data = X_train_tensor[:window_size*batch_size].view(batch_size, window_size, -1)
if torch.cuda.is_available() :
    init_data = init_data.to(device)

model = THOC(n_input, n_hidden, n_layers, n_centroids, init_data, tau=tau, dropout=dropout, cell_type=cell_type, batch_first=True)
if torch.cuda.is_available() :
    model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training
for epoch in range(num_epochs) :
    for i, window in enumerate(train_dl):
        if i == 0 :
            pass
        X, y = window
        X = X.to(device)
        y = y.to(device)

        anomaly_scores, centroids_diff, out_of_drnn = model.forward(X)

        loss = thoc_loss(anomaly_scores, centroids_diff, out_of_drnn, X)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0 :
            print("Epochs : %d, windows : %d  loss : %1.5f" %(epoch, i, loss.item()))


# test 할 때는 window size 다르게 실험한 뒤 average
# sliding window with size 80 and step 1
# 3-layer DRNN and lambda_l2reg = 10**(-6)
# n_hidden [32, 64, 84]
# cluster_centers [6/6/6 12/6/1 12/6/4 18/6/1 18/12/4 18/1286 32/12/6]
# lambda_orth, lambda_tss [0.01 0.1 1 10 100]
# Adam, lr = 0.01 -> step decay (0.65times per 20 epochs)
# tau [100, 1, 0.1, 0.05] -> step increase (0.67times per 5 epochs)
# run on GPU

# validation
threshold = torch.max(anomaly_scores)

def predict_normal(anomaly_scores) :

    normal = anomaly_scores < threshold
    normal = normal.long()
    pred_normal = normal.view(batch_size, window_size, -1)

    return pred_normal

def evaluate_model(pred, y) :

    TP_, TN_, FP_, FN_ = 0, 0, 0, 0
    for b in range(pred.shape[0]) :
        for t in range(pred.shape[1]) :
            for i in range(len(y)) :
                if (pred[b,t,i]==1) & (y[t,i]==1) :
                    TP_ += 1
                elif (pred[b,t,i]==0) & (y[t,i]==0) :
                    TN_ += 1
                elif (pred[b,t,i]==1) & (y[t,i]==0) :
                    FP_ += 1
                elif (pred[b,t,i]==0) & (y[t,i]==1) :
                    FN_ += 1
                else :
                    raise

    return TP_, TN_, FP_, FN_

TP, TN, FP, FN = 0, 0, 0, 0
for i, window in enumerate(valid_dl):
    with torch.no_grad() :
        X, y = window

        anomaly_scores, _, _ = model.forward(X)

        pred = predict_normal(anomaly_scores)

        TP, TN, FP, FN = evaluate_model(pred,y)

        TP += TP
        TN += TN
        FP += FP
        FN += FN


precision = TP / (TP+FP)
recall = TP / (TP+FN)
f1_score =  2*precision*recall / (precision+recall)