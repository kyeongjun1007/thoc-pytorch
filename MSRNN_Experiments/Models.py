import math
import torch
import torch.nn as nn


class VRNN (nn.RNN) :
    def __init__(self, n_input, n_hidden, n_out, n_layers, dropout = 0, nonlinearity='tanh', batch_first=True):
        super(VRNN, self).__init__(n_input, n_hidden, n_layers, nonlinearity=nonlinearity, batch_first=batch_first)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.linear = nn.Linear(n_hidden, n_out, device=self.device)

    def fit_dim(self, out):
        out_input_dim = self.linear(out)

        return out_input_dim


class VLSTM (nn.LSTM) :
    def __init__(self, n_input, n_hidden, n_out, n_layers, dropout = 0, batch_first=True):
        super(VLSTM, self).__init__(n_input, n_hidden, n_layers, batch_first=batch_first)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.linear = nn.Linear(n_hidden, n_out, device=self.device)

    def fit_dim(self, out):
        out_input_dim = self.linear(out)

        return out_input_dim


class VGRU (nn.GRU) :
    def __init__(self, n_input, n_hidden, n_out, n_layers, dropout = 0, batch_first=True):
        super(VGRU, self).__init__(n_input, n_hidden, n_layers, batch_first=batch_first)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.linear = nn.Linear(n_hidden, n_out, device=self.device)

    def fit_dim(self, out):
        out_input_dim = self.linear(out)

        return out_input_dim


class MSRNN(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers, dropout=0, cell_type='GRU', batch_first=True):
        super(MSRNN, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.input_scale = [2 ** i for i in range(n_layers)]  # scales of layers for additional input
        self.cell_type = cell_type  # cell type   (RNN, LSTM, GRU)
        self.batch_first = batch_first
        self.n_hidden = n_hidden

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

        # make multiscale inputs
        ms_inputs = [inputs.unfold(0, s, 1).mean(dim=3) for s in self.input_scale]

        for l, (cell, scale) in enumerate(zip(self.cells, self.input_scale)):  # l = layer number
            if hidden is None:
                inputs, _ = self._msrnn_layer(cell, inputs, ms_inputs[l], scale)
            else:
                inputs, hidden[l] = self._msrnn_layer(cell, inputs, ms_inputs[l], scale, hidden[l])

            if self.batch_first :
                layers.append(inputs.transpose(0, 1))
            else:
                layers.append(inputs)

        if self.batch_first :
            inputs = inputs.transpose(0, 1)

        return inputs, layers

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
    def __init__(self, n_input, n_hidden, n_ms_input, dropout, device, bias=True):
        super(MSRNNCell, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.device = device

        self.W_ih = nn.Linear(n_input, n_hidden, bias=bias,  device=self.device)  # input to hidden
        self.W_hh = nn.Linear(n_hidden, n_hidden, bias=bias,  device=self.device)  # hidden to hidden
        self.W_mh = nn.Linear(n_ms_input, n_hidden, bias=bias,  device=self.device)  # multi scale input to hidden
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
            # gates = gates.squeeze()
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

            # gate_x = gate_x.squeeze()
            # gate_h = gate_h.squeeze()
            # gate_n = gate_n.squeeze()

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


use_cuda = torch.cuda.is_available()


class DRNN(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers, dropout=0, cell_type='GRU', batch_first=False):
        super(DRNN, self).__init__()

        self.dilations = [2 ** i for i in range(n_layers)]
        self.cell_type = cell_type
        self.batch_first = batch_first

        layers = []
        if self.cell_type == "GRU":
            cell = nn.GRU
        elif self.cell_type == "RNN":
            cell = nn.RNN
        elif self.cell_type == "LSTM":
            cell = nn.LSTM
        else:
            raise NotImplementedError

        for i in range(n_layers):
            if i == 0:
                c = cell(n_input, n_hidden, dropout=dropout)
            else:
                c = cell(n_hidden, n_hidden, dropout=dropout)
            layers.append(c)
        self.cells = nn.Sequential(*layers)

    def forward(self, inputs, hidden=None):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        outputs = []
        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if hidden is None:
                inputs, _ = self.drnn_layer(cell, inputs, dilation)
            else:
                inputs, hidden[i] = self.drnn_layer(cell, inputs, dilation, hidden[i])

            outputs.append(inputs[-dilation:])

        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        return inputs, outputs

    def drnn_layer(self, cell, inputs, rate, hidden=None):
        n_steps = len(inputs)
        batch_size = inputs[0].size(0)
        hidden_size = cell.hidden_size

        inputs, _ = self._pad_inputs(inputs, n_steps, rate)
        dilated_inputs = self._prepare_inputs(inputs, rate)

        if hidden is None:
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)
        else:
            hidden = self._prepare_inputs(hidden, rate)
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size, hidden=hidden)

        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        outputs = self._unpad_outputs(splitted_outputs, n_steps)

        return outputs, hidden

    def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size, hidden=None):
        if hidden is None:
            if self.cell_type == 'LSTM':
                c, m = self.init_hidden(batch_size * rate, hidden_size)
                hidden = (c.unsqueeze(0), m.unsqueeze(0))
            else:
                hidden = self.init_hidden(batch_size * rate, hidden_size).unsqueeze(0)

        dilated_outputs, hidden = cell(dilated_inputs, hidden)

        return dilated_outputs, hidden

    def _unpad_outputs(self, splitted_outputs, n_steps):
        return splitted_outputs[:n_steps]

    def _split_outputs(self, dilated_outputs, rate):
        batchsize = dilated_outputs.size(1) // rate

        blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :] for i in range(rate)]

        interleaved = torch.stack((blocks)).transpose(1, 0).contiguous()
        interleaved = interleaved.view(dilated_outputs.size(0) * rate,
                                       batchsize,
                                       dilated_outputs.size(2))
        return interleaved

    def _pad_inputs(self, inputs, n_steps, rate):
        is_even = (n_steps % rate) == 0

        if not is_even:
            dilated_steps = n_steps // rate + 1

            zeros_ = torch.zeros(dilated_steps * rate - inputs.size(0),
                                 inputs.size(1),
                                 inputs.size(2))
            if use_cuda:
                zeros_ = zeros_.cuda()

            inputs = torch.cat((inputs, zeros_))
        else:
            dilated_steps = n_steps // rate

        return inputs, dilated_steps

    def _prepare_inputs(self, inputs, rate):
        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)
        return dilated_inputs

    def init_hidden(self, batch_size, hidden_dim):
        hidden = torch.zeros(batch_size, hidden_dim)
        if use_cuda:
            hidden = hidden.cuda()
        if self.cell_type == "LSTM":
            memory = torch.zeros(batch_size, hidden_dim)
            if use_cuda:
                memory = memory.cuda()
            return (hidden, memory)
        else:
            return hidden