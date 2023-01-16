import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available()


class VRNN(nn.Module) :
    def __init__(self, n_input, n_hidden, n_layers, nonlinearity = 'tanh', batch_first = 'True'):
        super(VRNN, self).__init__()
        if use_cuda:
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        self.cells = nn.ModuleList([VRNNCell(n_input, n_hidden, self.device)] +
                                   [VRNNCell(n_hidden, n_hidden, self.device) for _ in range(n_layers - 1)])
        self.linear = nn.Linear(n_hidden, n_input, device=self.device)
        self.batch_first = batch_first
        self.nonlinearity = nonlinearity
        self.n_hidden = n_hidden

    def forward(self, inputs):
        if self.batch_first :
            inputs = inputs.transpose(0,1)

        out_all = []

        for l in range(len(self.cells)) :
            if l == 0 :
                out = self.apply_layer(inputs, l)
            else :
                out = self.apply_layer(out, l)
            out_all.append(self.linear(out))

        return out_all

    def apply_layer(self, input, l) :
        out = []
        cell = self.cells[l]
        for i in range(len(input)) :
            if i == 0 :
                hidden = torch.randn(input.shape[1], self.n_hidden, device=self.device)
            if self.nonlinearity == 'tanh' :
                hidden = torch.tanh(cell.gate_i(input[i]) + cell.gate_h(hidden))
            elif self.nonlinearity == 'relu' :
                hidden = torch.relu(cell.gate_i(input[i]) + cell.gate_h(hidden))

            out.append(hidden)
        out = torch.stack(out)

        return out


class VRNNCell(nn.Module):

    def __init__(self, input_size, hidden_size, device):
        super(VRNNCell, self).__init__()
        self.gate_i = nn.Linear(input_size, hidden_size, device=device)
        self.gate_h = nn.Linear(hidden_size, hidden_size, device=device)