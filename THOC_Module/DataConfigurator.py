import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from pickle import dump  # save scaler

class DataConfigurator :
    def __init__(self, train_params) :
        self.device = torch.device("cuda", train_params['cuda_device_num'] if train_params['use_cuda'] else "cpu")

        self.filename = train_params['data_config']['file_name'][:-4]
        self.header = train_params['data_config']['header']
        self.index_col = train_params['data_config']['index_col']
        self.data = pd.read_csv(self.filename+'.csv', header=self.header, index_col=self.index_col)

        self.window_size = train_params['window_size']
        self.batch_size = train_params['batch_size']
        self.train_len = int(len(self.data)*(1-train_params['data_config']['valid_ratio'] - train_params['data_config']['test_ratio']))
        self.valid_len = int(len(self.data)*(train_params['data_config']['valid_ratio']))
        self.test_len = int(len(self.data)*(train_params['data_config']['test_ratio']))
        self.shuffle = train_params['data_config']['shuffle']
        self.scaler = MinMaxScaler()

        self._fit_scaler()
        self._save_test_set()

    def _fit_scaler(self) :
        train = self.data.iloc[:self.train_len, :]
        self.scaler.fit(train)

        #!! path 설정
        dump(self.scaler, open('./scaler.pkl', 'wb'))  # save scaler as pickle

    def train_dataloader(self) :
        train = self.data.iloc[:self.train_len, :]
        train = self.scaler.transform(train)
        train_tensor = torch.tensor((np.array(train, dtype='float32')))
        train_dataset = SlidingWindow(X=train_tensor, window=self.window_size)
        train_dl = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=self.shuffle).to(self.device)

        return train_dl

    def valid_dataloader(self) :
        start_index = self.train_len
        valid = self.data.iloc[start_index:start_index+self.valid_len, :]
        valid_tensor = torch.tensor((np.array(valid, dtype='float32')))
        valid_dataset = SlidingWindow(X=valid_tensor, window=self.window_size)
        valid_dl = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=self.shuffle).to(self.device)

        return valid_dl

    def _save_test_set(self) :
        start_index = self.train_len + self.valid_len
        test = self.data.iloc[start_index:start_index+self.test_len, :]

        #!! path 설정
        test.to_csv(self.filename + '_test.csv', header=self.header, index=self.index_col)

    def get_init_data(self) :
        init = self.data.iloc[:self.window_size*self.batch_size]
        init = self.scaler.transform(init)
        init_tensor = torch.tensor((np.array(init, dtype='float32')))
        init_data = init_tensor.view(self.batch_size, self.window_size, -1).to(self.device)

        return init_data


class SlidingWindow(Dataset):
    def __init__(self, X, window):
        self.X = X
        self.window = window

    def __getitem__(self, index):
        out_X = self.X[index:index + self.window]
        return out_X

    def __len__(self):
        return len(self.X) - self.window