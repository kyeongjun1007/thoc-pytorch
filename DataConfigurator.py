import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from pickle import dump  # save scaler


class DataConfigurator:
    def __init__(self, train_params, result_folder):
        self.device = torch.device("cuda", train_params['cuda_device_num'] if train_params['use_cuda'] else "cpu")

        self.dataname = '/' + train_params['data_config']['data_name'][:-4]
        self.header = train_params['data_config']['header']
        self.index_col = train_params['data_config']['index_col']
        self.data = pd.read_csv(train_params['data_config']['data_dir'] + train_params['data_config']['data_name'],
                                header=self.header, index_col=self.index_col)
        if train_params['data_config']['label']:
            self.label = self.data.iloc[:, -1]
            self.data = self.data.iloc[:, :-1]
        else:
            self.label = None

        self.window_size = train_params['window_size']
        self.batch_size = train_params['batch_size']
        self.train_len = int(len(self.data) * (
                    1 - train_params['data_config']['valid_ratio'] - train_params['data_config']['test_ratio']))
        self.valid_len = int(len(self.data) * (train_params['data_config']['valid_ratio']))
        self.test_len = int(len(self.data) * (train_params['data_config']['test_ratio']))
        self.shuffle = train_params['data_config']['shuffle']
        self.scaler = MinMaxScaler()

        self._fit_scaler(result_folder)
        self._save_test_set(result_folder)

    def _fit_scaler(self, result_folder):
        train = self.data.iloc[:self.train_len, :]
        self.scaler.fit(train)

        dump(self.scaler, open(f'{result_folder}/scaler.pkl', 'wb'))  # save scaler as pickle

    def train_dataloader(self):
        train = self.data.iloc[:self.train_len, :]
        train = self.scaler.transform(train)
        train_tensor = torch.tensor((np.array(train, dtype='float32')))
        train_dataset = SlidingWindow(X=train_tensor, window=self.window_size)
        train_dl = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        return train_dl

    def valid_dataloader(self):
        start_index = self.train_len
        valid = self.data.iloc[start_index:start_index + self.valid_len, :]
        valid = self.scaler.transform(valid)
        valid_tensor = torch.tensor((np.array(valid, dtype='float32')))
        valid_dataset = SlidingWindow(X=valid_tensor, window=self.window_size)
        valid_dl = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=False)

        return valid_dl

    def _save_test_set(self, result_folder):
        start_index = self.train_len + self.valid_len
        test = self.data.iloc[start_index:start_index + self.test_len, :]

        if self.label is not None:
            test['label'] = self.label.iloc[start_index:start_index + self.test_len]

        test.to_csv(result_folder + self.dataname + '_test.csv', header=False if self.header is None else True,
                    index=self.index_col)

    def get_init_data(self, use_cuda):
        init = self.data.iloc[:self.window_size * self.batch_size]
        init = self.scaler.transform(init)
        init_tensor = torch.tensor((np.array(init, dtype='float32')))
        if use_cuda:
            init_data = init_tensor.view(self.batch_size, self.window_size, -1).to(self.device)
        else:
            init_data = init_tensor.view(self.window_size, self.batch_size, -1).to(self.device)

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


class SlidingWindowTest(Dataset):
    def __init__(self, X, y, window):
        self.X = X
        self.y = y
        self.window = window

    def __getitem__(self, index):
        out_X = self.X[index:index + self.window]
        out_y = self.y[index:index + self.window]
        return out_X, out_y

    def __len__(self):
        return len(self.X) - self.window
