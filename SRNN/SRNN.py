import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

use_cuda = torch.cuda.is_available()

class SRNN(nn.Module) :

    def __init__(self, n_input, n_hidden, n_layers, nonlinearity = 'tanh', batch_first = 'True') :
        super(SRNN, self).__init__()
        if use_cuda :
            self.device = "cuda:0"
        else :
            self.device = "cpu"
        self.cells = nn.ModuleList([SRNNCell(n_input, n_hidden, n_input, nonlinearity=nonlinearity, device = self.device)] +
                                   [SRNNCell(n_hidden, n_hidden, n_input, nonlinearity=nonlinearity, device = self.device) for _ in range(n_layers-1)])
        self.scales = [2 ** i for i in range(n_layers)]
        self.batch_first = batch_first

    def forward(self, inputs) :
        if self.batch_first :
            inputs = inputs.transpose(0,1)

        out_all = []
        for l in range(len(self.cells)) :
            scaled_x = self.scaled_input(inputs, self.scales[l])
            if l == 0 :
                out = self. apply_layer(inputs, scaled_x, l)
            else :
                out = self.apply_layer(out, scaled_x, l)
            out_all.append(out)

        return out_all

    def scaled_input(self, x, scale) :
        scaled_x = x.unfold(0, scale, 1)
        scaled_x = torch.mean(scaled_x, 3)
        return scaled_x

    def apply_layer(self, input, scaled_x, l) :
        out = []
        cell = self.cells[l]
        for i in range(scaled_x.shape[0]-self.scales[l]) :
            index = self.scales[l]-1+i
            if i == 0 :
                hidden = cell.apply_cell(input[index], scaled_x[index])
            else :
                hidden = cell.apply_cell(input[index], scaled_x[index], hidden)
            out.append(hidden)

        out = torch.stack(out)

        return out


class SRNNCell(nn.Module) :

    def __init__(self, input_size, hidden_size, scaled_size, nonlinearity, device = 'cpu', dtype = None):
        super(SRNNCell, self).__init__()
        factory_kwargs = {'device':device, 'dtype': dtype}
        self.n_hidden = hidden_size
        self.nonlinearity = nonlinearity
        self.weight_i = Parameter(torch.empty((input_size, hidden_size), **factory_kwargs))
        self.weight_s = Parameter(torch.empty((scaled_size, hidden_size), **factory_kwargs))
        self.weight_h = Parameter(torch.empty((hidden_size, hidden_size), **factory_kwargs))
        # self.bias_i = Parameter(torch.empty(hidden_size, **factory_kwargs))
        # self.bias_s = Parameter(torch.empty(hidden_size, **factory_kwargs))
        # self.bias_h = Parameter(torch.empty(hidden_size, **factory_kwargs))

    def apply_cell(self, x, scaled_x, hidden = None) :
        if hidden is None:
            if use_cuda :
                hidden = torch.empty(x.shape[0], self.n_hidden).cuda()
            else :
                hidden = torch.empty(x.shape[0], self.n_hidden)
        if self.nonlinearity == 'tanh' :
            hidden = torch.tanh(torch.matmul(x, self.weight_i) +
                                torch.matmul(hidden, self.weight_h) +
                                torch.matmul(scaled_x, self.weight_s))
        elif self.nonlinearity == 'relu' :
            hidden = torch.relu(torch.matmul(x, self.weight_i) +
                                torch.matmul(hidden, self.weight_h) +
                                torch.matmul(scaled_x, self.weight_s))
        return hidden