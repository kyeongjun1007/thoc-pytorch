from keras.datasets import mnist
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from Models import VRNN, VLSTM, VGRU, MSRNN, DRNN


(train_x, train_y), _ = mnist.load_data()
train_x = np.reshape(train_x, (train_x.shape[0], -1))
train_y = np.reshape(train_y, (train_y.shape[0], -1))

valid_x = train_x[50000:]
valid_y = train_y[50000:]
train_x = train_x[:50000]
train_y = train_y[:50000]


def get_data_loader(batch_size):
    train_loader = DataLoader(list(zip(train_x, train_y)), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(list(zip(valid_x, valid_y)), batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader


class VRNN_Copy(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, cell_type, dropout, out_size, batch_first):
        super(VRNN_Copy, self).__init__()
        if cell_type == 'RNN':
            self.vrnn = VRNN(dropout=dropout, n_hidden=hidden_size, n_out=out_size,
                             n_input=input_size, n_layers=num_layers, batch_first=batch_first)
        elif cell_type == 'LSTM':
            self.vrnn = VLSTM(dropout=dropout, n_hidden=hidden_size, n_out=out_size,
                              n_input=input_size, n_layers=num_layers, batch_first=batch_first)
        elif cell_type == 'GRU':
            self.vrnn = VGRU(dropout=dropout, n_hidden=hidden_size, n_out=out_size,
                             n_input=input_size, n_layers=num_layers, batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, out_size)
        self._init_weights()

    def _init_weights(self):
        self.linear.weight.data.normal_(0,0.01)

    def forward(self, x): # x: (batch, steps, input_size)
        y, _ = self.vrnn(x) # y: (batch, steps, hidden_size)

        return self.linear(y[:,-1]).unsqueeze(1) # (batch, steps, output_size)


class MSRNN_Copy(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, cell_type, dropout, out_size, batch_first):
        super(MSRNN_Copy, self).__init__()
        self.msrnn = MSRNN(cell_type=cell_type, dropout=dropout, n_hidden=hidden_size,
                           n_input=input_size, n_layers=num_layers, batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, out_size)
        self._init_weights()

    def _init_weights(self):
        self.linear.weight.data.normal_(0,0.01)

    def forward(self, x): # x: (batch, steps, input_size)
        y, _ = self.msrnn(x) # y: (batch, steps, hidden_size)

        return self.linear(y[:,-1]).unsqueeze(1) # (batch, steps, out_size)


class DRNN_Copy(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, cell_type, dropout, out_size, batch_first):
        super(DRNN_Copy, self).__init__()
        self.drnn = DRNN(cell_type=cell_type, dropout=dropout, n_hidden=hidden_size,
                         n_input=input_size, n_layers=num_layers, batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, out_size)
        self._init_weights()

    def _init_weights(self):
        self.linear.weight.data.normal_(0,0.01)

    def forward(self, x): # x: (batch, steps, input_size)
        y, _ = self.drnn(x) # y: (batch, steps, hidden_size)

        return self.linear(y[:,-1]).unsqueeze(1) # (batch, steps, output_size)