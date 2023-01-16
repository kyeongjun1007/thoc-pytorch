from VRNN_s import VRNN
import torch
import torch.nn as nn
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
path = "./gridsearch/vrnn_s"
if not Path(path).exists():
    Path(path).mkdir(parents=True)

## logger config
logging.basicConfig(level=logging.INFO, filename=path + "/param_grid.log", filemode="w", format="%(message)s")

## data split and device setting
data = pd.read_csv("PowerDemand.csv")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train, other = train_test_split(data, test_size=0.5, shuffle=False)
valid, test = train_test_split(other, test_size=0.4, shuffle=False)

train_X, train_y = train[:-1], train[1:]
valid_X, valid_y = valid[:-1], valid[1:]

## scaling and save train scaler
X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
train_X = X_scaler.fit_transform(train_X)
train_y = y_scaler.fit_transform(train_y)
valid_X = X_scaler.transform(valid_X)

dump(X_scaler, open(path + '/x_scaler.pkl', 'wb'))  # save scaler as pickle
dump(y_scaler, open(path + '/y_scaler.pkl', 'wb'))

## set datatype to float
train_X = torch.tensor((np.array(train_X, dtype="float32")))
train_y = torch.tensor((np.array(train_y, dtype="float32")))
valid_X = torch.tensor((np.array(valid_X, dtype="float32")))
valid_y = torch.tensor((np.array(valid_y, dtype="float32")))


# sliding window
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


# loss function
def mse_loss(out, y):

    loss = ((out - y) ** 2).mean()

    return loss


# parameters config
## fixed
n_input = 1
nonlinearity = 'tanh'
window_size = 80

## grid-search
n_hidden_ = [16, 32, 64, 128]
n_layers_ = [3, 4, 5]
batch_size_ = [1024, 512, 128]
num_epochs_ = [5, 10, 15, 20]
learning_rate_ = [1e-02, 1e-03]
weight_decay_ = [1e-05, 1e-06]

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
                        train_dataset = SlidingWindow(X=train_X, y=train_y, window=window_size)
                        valid_dataset = SlidingWindow(X=valid_X, y=valid_y, window=window_size)

                        train_dataloader = DataLoader(dataset=train_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=False)
                        valid_dataloader = DataLoader(dataset=valid_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=False)

                        model = VRNN(n_input, n_hidden, n_layers, nonlinearity=nonlinearity).to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

                        # training
                        for epoch in range(num_epochs) :
                            for i, data_ in enumerate(train_dataloader) :
                                window, y = data_[0], data_[1]
                                window = window.to(device)
                                y = y[:,-1,:].to(device)

                                _out = model.forward(window)[0][:,-1,:]
                                out = model.fit_dim(_out)

                                loss = mse_loss(out, y)

                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()

                        # validation
                        out_list, y_list = [], []
                        for i, data_ in enumerate(valid_dataloader) :
                            window, y = data_[0], data_[1]
                            window = window.to(device)

                            _out = model.forward(window)[0][:,-1,:]
                            out = model.fit_dim(_out)

                            out_list.append(y_scaler.inverse_transform(out.cpu().detach().numpy()))

                        hat_y = np.concatenate(out_list)
                        hat_y = hat_y.flatten()

                        real_y = valid_y[80:].numpy().flatten()

                        mse = ((np.subtract(hat_y, real_y)) ** 2).mean()

                        logging.info(f"\t\t\t\t\ttraining loss : {loss.item()}")
                        logging.info(f"\t\t\t\t\tvalidation loss : {mse}\n")
                        count += 1

                        # make parameter dictionary
                        param_dict[n_layers][n_hidden][batch_size][num_epochs][learning_rate][weight_decay] = mse

                        # save best model
                        if top_loss > mse :
                            torch.save(model.state_dict(), path + "/model_state_dict.pt")
                            top_loss = mse

                        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                              "experiment " + str(count) + "/" + str(total) + " done..." )
                        print("validation loss : %d, best : %d" %(mse, top_loss))

with open (path + 'param_dict.pkl','wb') as f :
    pickle.dump(param_dict,f)