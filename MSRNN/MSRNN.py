import math
import torch
import torch.nn as nn


class MSRNN(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers, dropout=0, cell_type='GRU', batch_first=True):
        super(MSRNN, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.input_scale = [2 ** i for i in range(n_layers)]  # scales of layers for additional input
        self.cell_type = cell_type  # cell type   (RNN, LSTM, GRU)
        self.batch_first = batch_first
        self.n_hidden = n_hidden
        self.linear = nn.Linear(n_hidden, n_input, device=self.device)  # linear layer for output

        if self.cell_type == "GRU":
            cell = MSGRUCell
        elif self.cell_type == "RNN":
            cell = MSRNNCell
        elif self.cell_type == "LSTM":
            cell = MSLSTMCell
        else:
            raise NotImplementedError

        layers = [cell(n_input if i == 0 else n_hidden, n_hidden, n_input, dropout=dropout, device=self.device)
                  for i in range(n_layers)]

        self.cells = nn.Sequential(*layers)

    def forward(self, inputs, hidden=None):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)

        layers = []
        hiddens = []

        # make multiscale inputs
        ms_inputs = [inputs.unfold(0, s, 1).mean(dim=3) for s in self.input_scale]

        for l, (cell, scale) in enumerate(zip(self.cells, self.input_scale)):  # l = layer number
            if hidden is None:
                inputs, k = self._msrnn_layer(cell, inputs, ms_inputs[l], scale)
            else:
                inputs, hidden[l] = self._msrnn_layer(cell, inputs, ms_inputs[l], scale, hidden[l])
                k = hidden[l]

            layers.append(inputs)
            hiddens.append(k)

        # linear layer for output (n_layers, batch_size, seq_len, n_hidden) -> (n_layers, batch_size, seq_len, n_input)
        layers = [self.linear(layer) for layer in layers]

        return layers, hiddens

    def _msrnn_layer(self, cell, inputs, ms_input, scale, hidden=None):
        batch_size = inputs[0].size(0)
        hidden_size = cell.n_hidden

        if hidden is None:
            if self.cell_type == 'LSTM':
                c, m = self._init_hidden(batch_size, hidden_size)
                hidden = (c, m)
            else:
                hidden = self._init_hidden(batch_size, hidden_size)

        outputs, hidden = cell(inputs, ms_input, hidden)

        return outputs, hidden

    def _init_hidden(self, batch_size, hidden_size):
        if self.cell_type == "LSTM":
            return (torch.zeros(batch_size, hidden_size, device=self.device),
                    torch.zeros(batch_size, hidden_size, device=self.device))
        else:
            return torch.zeros(batch_size, hidden_size, device=self.device)


# multi scale RNN cell
class MSRNNCell(nn.Module):
    def __init__(self, n_input, n_hidden, n_ms_input, dropout, device):
        super(MSRNNCell, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.device = device

        self.W_ih = nn.Linear(n_input, n_hidden, device=self.device)  # input to hidden
        self.W_hh = nn.Linear(n_hidden, n_hidden, device=self.device)  # hidden to hidden
        self.W_mh = nn.Linear(n_ms_input, n_hidden, device=self.device)  # multi scale input to hidden
        # self.W_ho = nn.Linear(n_hidden, n_hidden, device=self.device)   # hidden to output

    def forward(self, inputs, ms_inputs, hidden):
        # inputs.shape = (input_length, batch_size, input_dim)
        # hidden.shape = (batch_size, n_hidden)
        # outputs.shape = (input_length, batch_size, n_hidden)

        # apply multi scale RNN cell
        outputs = []
        for i in range(ms_inputs.size(0)):
            hidden = torch.tanh(self.W_ih(inputs[inputs.size(0)-ms_inputs.size(0)+i]) + self.W_hh(hidden) + self.W_mh(ms_inputs[i]))
            outputs.append(hidden)

        outputs = torch.stack(outputs)
        return outputs, hidden


# multi scale LSTM cell
class MSLSTMCell(nn.Module):
    def __init__(self, n_input, n_hidden, n_ms_input, dropout, device, bias=True):
        super(MSLSTMCell, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.dropout = dropout

        self.W_ih = nn.Linear(n_input, 4 * n_hidden, bias=bias, device=device)             # input to hidden (devided to 4 gates)
        self.W_hh = nn.Linear(n_hidden, 4 * n_hidden, bias=bias, device=device)            # hidden to hidden (devided to 4 gates)
        self.W_nh = nn.Linear(n_ms_input, 4 * n_hidden, bias=bias, device=device)          # multiscale input to hidden (devided to 4 gates)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.n_hidden)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, inputs, ms_inputs,  hidden):
        # inputs.shape = (input_length, batch_size, input_dim)
        # hidden.shape = (batch_size, n_hidden)
        # outputs.shape = (input_length, batch_size, n_hidden)

        hx, cx = hidden

        outs = []
        for i in range(ms_inputs.size(0)):
            input = inputs[inputs.size(0)-ms_inputs.size(0)+i]
            ms_input = ms_inputs[i]

            input = input.view(-1, input.size(1))
            ms_input = ms_input.view(-1, ms_input.size(1))

            gates = self.W_ih(input) + self.W_hh(hx) + self.W_nh(ms_input)
            gates = gates.squeeze()
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cx = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
            hx = torch.mul(outgate, torch.tanh(cx))

            outs.append(hx)

        outs = torch.stack(outs)

        return outs, (hx, cx)


# multi scale GRU cell
class MSGRUCell(nn.Module):
    def __init__(self, n_input, n_hidden, n_ms_input, dropout, device, bias=True):
        super(MSGRUCell, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.bias = bias
        self.dropout = dropout

        self.xh = nn.Linear(n_input, 3 * n_hidden, bias=bias, device=device)
        self.hh = nn.Linear(n_hidden, 3 * n_hidden, bias=bias, device=device)
        self.nh = nn.Linear(n_ms_input, 3 * n_hidden, bias=bias, device=device)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.n_hidden)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, inputs,ms_inputs, hidden):
        # inputs.shape = (input_length, batch_size, input_dim)
        # hidden.shape = (batch_size, n_hidden)
        # outputs.shape = (input_length, batch_size, n_hidden)

        outs = []
        for i in range(ms_inputs.size(0)):
            input = inputs[inputs.size(0)-ms_inputs.size(0)+i]
            ms_input = ms_inputs[i]

            input = input.view(-1, input.size(1))
            ms_input = ms_input.view(-1, ms_input.size(1))

            gate_x = self.xh(input)
            gate_h = self.hh(hidden)
            gate_n = self.nh(ms_input)

            gate_x = gate_x.squeeze()
            gate_h = gate_h.squeeze()
            gate_n = gate_n.squeeze()

            i_r, i_i, i_n = gate_x.chunk(3, 1)
            h_r, h_i, h_n = gate_h.chunk(3, 1)
            n_r, n_i, n_n = gate_n.chunk(3, 1)

            resetgate = torch.sigmoid(i_r + h_r + n_r)
            inputgate = torch.sigmoid(i_i + h_i + n_i)
            newgate = torch.tanh(i_n + (resetgate * h_n) + n_n)

            hidden = newgate + inputgate * (hidden - newgate)
            outs.append(hidden)

        outs = torch.stack(outs)

        return outs, hidden
