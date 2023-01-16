import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available()


class SRNN(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers, nonlinearity='tanh', batch_first='True'):
        super(SRNN, self).__init__()
        if use_cuda:
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        self.cells = nn.ModuleList([SRNNCell(n_input, n_hidden, n_input, self.device)] +
                                   [SRNNCell(n_hidden, n_hidden, n_input, self.device) for _ in range(n_layers - 1)])
        self.linear = nn.Linear(n_hidden, n_input, device=self.device)
        self.scales = [2 ** i for i in range(n_layers)]
        self.batch_first = batch_first
        self.nonlinearity = nonlinearity
        self.n_hidden = n_hidden

    def forward(self, inputs):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)

        out_all = []
        scaled_x_all = []
        for l in range(len(self.cells)):
            scaled_x = self.scaled_input(inputs, self.scales[l])
            if l == 0:
                out = self.apply_layer(inputs, scaled_x, l)
            else:
                out = self.apply_layer(out, scaled_x, l)
            out_all.append(self.linear(out))
            scaled_x_all.append(scaled_x)

        return out_all, scaled_x_all

    def scaled_input(self, x, scale):
        scaled_x = x.unfold(0, scale, 1)
        scaled_x = scaled_x.mean(dim=3)
        return scaled_x

    def apply_layer(self, input, scaled_x, l):
        out = []
        cell = self.cells[l]
        for i in range(scaled_x.shape[0]):
            if l == 0:
                index = self.scales[l] - 1 + i
            else:
                index = self.scales[l] - self.scales[l - 1] - 1 + i
            if i == 0:
                hidden = torch.randn(input.shape[1], self.n_hidden, device=self.device)
            if self.nonlinearity == 'tanh':
                hidden = torch.tanh(cell.gate_i(input[index]) + cell.gate_s(scaled_x[i]) + cell.gate_h(hidden))
            elif self.nonlinearity == 'relu':
                hidden = torch.relu(cell.gate_i(input[index]) + cell.gate_s(scaled_x[i]) + cell.gate_h(hidden))

            out.append(hidden)

        out = torch.stack(out)

        return out


class SRNNCell(nn.Module):

    def __init__(self, input_size, hidden_size, scaled_size, device):
        super(SRNNCell, self).__init__()
        self.n_hidden = hidden_size
        self.gate_i = nn.Linear(input_size, hidden_size, device=device)
        self.gate_s = nn.Linear(scaled_size, hidden_size, device=device)
        self.gate_h = nn.Linear(hidden_size, hidden_size, device=device)
