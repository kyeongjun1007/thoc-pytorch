from VRNN_m import VRNN
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
import datetime  # logging with datetime
import pickle  # save parameter dictionary
from pickle import dump  # save scaler
from pathlib import Path

# data config
## make directory
path = "./gridsearch/vrnn_m"
if not Path(path).exists():
    Path(path).mkdir(parents=True)

## logger config
logging.basicConfig(level=logging.INFO, filename=path + "/param_grid.log", filemode="w", format="%(message)s")

## data split and device setting
data = pd.read_csv("PowerDemand.csv")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train, other = train_test_split(data, test_size=0.5, shuffle=False)
valid, test = train_test_split(other, test_size=0.4, shuffle=False)

## scaling and save train scaler
scaler = MinMaxScaler()
train = scaler.fit_transform(train)
valid = scaler.transform(valid)

dump(scaler, open(path + '/scaler.pkl', 'wb'))  # save scaler as pickle

## set datatype to float
train = torch.tensor((np.array(train, dtype="float32")))
valid = torch.tensor((np.array(valid, dtype="float32")))


# sliding window
class SlidingWindow(Dataset):
    def __init__(self, X, window, n_layers):
        self.X = X
        self.window = window
        self.n_layer = n_layers

    def __getitem__(self, index):
        x_out = self.X[index:index + self.window]
        y_out = self.X[index+self.window : index+self.window + 2**(self.n_layer-1)]
        return x_out, y_out

    def __len__(self):
        return len(self.X) - self.window


def mse_loss_m(out, y):
    loss_list = []
    for i in range(len(out)) :
        loss_list.append((out[i][-1] - y[i])**2)
    loss = torch.stack(loss_list)
    loss = loss.mean()

    return loss

# parameters config
## fixed
n_input = 1
nonlinearity = 'tanh'
window_size = 80

## grid-search
n_hidden_ = [16, 32, 64, 128]
n_layers_ = [3, 4, 5]
batch_size_ = [128, 256, 512]
num_epochs_ = [5, 10, 15, 20]
learning_rate_ = [1e-02, 1e-03]
weight_decay_ = [1e-05, 1e-06]

## make scaled_y
real_y = []
for l in range(max(n_layers_)) :
    real_y_l = []
    for j in range(len(valid)-(2**l - 1)) :
        real_y_l.append(scaler.inverse_transform(valid)[j:j+2**l].mean())
    real_y.append(np.array(real_y_l))

# run experiment
top_loss = 9999
count = 0
total = len(n_hidden_) * len(n_layers_) * len(batch_size_) * len(num_epochs_) * len(learning_rate_) * len(weight_decay_)

param_dict = {}
for n_layers in n_layers_ :
    logging.info(f"n_layers : {n_layers}")
    param_dict[n_layers] = {}
    for n_hidden in n_hidden_ :
        logging.info(f"\tn_hidden : {n_hidden}")
        param_dict[n_layers][n_hidden] = {}
        for batch_size in batch_size_ :
            logging.info(f"\t\tbatch_size : {batch_size}")
            param_dict[n_layers][n_hidden][batch_size] = {}
            for num_epochs in num_epochs_ :
                logging.info(f"\t\t\tnum_epochs : {num_epochs}")
                param_dict[n_layers][n_hidden][batch_size][num_epochs] = {}
                for learning_rate in learning_rate_ :
                    logging.info(f"\t\t\t\tlearning_rate : {learning_rate}")
                    param_dict[n_layers][n_hidden][batch_size][num_epochs][learning_rate] = {}
                    for weight_decay in weight_decay_ :
                        logging.info(f"\t\t\t\tweight_decay : {weight_decay}")

                        train_dataset = SlidingWindow(X=train, window=window_size, n_layers=n_layers)
                        valid_dataset = SlidingWindow(X=valid, window=window_size, n_layers=n_layers)

                        train_dataloader = DataLoader(dataset=train_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=False)
                        valid_dataloader = DataLoader(dataset=valid_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=False)

                        model = VRNN(n_input, n_hidden, n_layers, nonlinearity=nonlinearity)
                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

                        # training
                        for epoch in range(num_epochs):
                            for i, data_ in enumerate(train_dataloader):
                                window, y = data_[0], data_[1]
                                window = window.to(device)
                                y = y.to(device)
                                y_list = [y[:,:2**k].mean(dim=1) for k in range(n_layers)]

                                out = model.forward(window)

                                loss = mse_loss_m(out, y_list)

                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                                if i == (len(train) - window_size - 2**(n_layers-1))//batch_size-1 :
                                    break

                        # validation
                        out_list = []
                        for i, data_ in enumerate(valid_dataloader):
                            window, y = data_[0], data_[1]
                            window = window.to(device)

                            out = model.forward(window)

                            out_list.append([scaler.inverse_transform(out[k][-1].cpu().detach().numpy()) for k in range(len(out))])
                            if i == (len(valid) - window_size - 2**(n_layers-1))//batch_size-1 :
                                break

                        hat_y = np.concatenate(out_list)
                        hat_y = [hat_y[k::n_layers].flatten() for k in range(n_layers)]

                        mse = 0
                        for i in range(len(hat_y)) :
                            mse += ((np.subtract(hat_y[i], real_y[i][window_size:window_size+len(hat_y[i])])) ** 2).mean()

                        logging.info(f"\t\t\t\t\ttraining loss : {loss.item()}")
                        logging.info(f"\t\t\t\t\tvalidation loss : {mse}\n")
                        count += 1

                        # make parameter dictionary
                        param_dict[n_layers][n_hidden][batch_size][num_epochs][learning_rate][weight_decay] = mse

                        # save best model
                        if top_loss > mse:
                            torch.save(model.state_dict(), path + "/model_state_dict.pt")
                            top_loss = mse

                        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                              "experiment " + str(count) + "/" + str(total) + " done...")
                        print("validation loss : %d, best : %d" % (mse, top_loss))

with open (path + 'param_dict.pkl','wb') as f :
    pickle.dump(param_dict,f)