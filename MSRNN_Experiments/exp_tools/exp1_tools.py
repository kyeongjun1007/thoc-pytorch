import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from Models import VRNN, VLSTM, VGRU, MSRNN, DRNN


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

        return self.linear(y) # (batch, steps, output_size)


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

        return self.linear(y[-1]) # (batch, steps, out_size)


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

        return self.linear(y) # (batch, steps, output_size)


def data_generator(T, mem_length):

    seq = torch.from_numpy(np.random.randint(0, 8, size=(mem_length,))).float()
    blanck = 8 * torch.ones(T)
    marker = 9 * torch.ones(mem_length + 1)
    placeholders = 8 * torch.ones(mem_length)

    x = torch.cat((seq, blanck[:-1], marker), 0)
    y = torch.cat((placeholders, blanck, seq), 0).long()

    x, y = Variable(x), Variable(y)
    return x.unsqueeze(0), y.unsqueeze(0)