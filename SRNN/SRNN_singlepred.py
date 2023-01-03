import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available()

class SRNN(nn.Module) :

    def __init__(self, n_input, n_hidden, n_layers, nonlinearity = 'tanh', batch_first = 'True') :
        super(SRNN, self).__init__()
        if use_cuda :
            self.device = "cuda:0"
        else :
            self.device = "cpu"
        self.cells = nn.ModuleList([SRNNCell(n_input, n_hidden, n_input, self.device)] +
                                   [SRNNCell(n_hidden, n_hidden, n_input, self.device) for _ in range(n_layers-1)])
        self.linear = nn.Linear(n_hidden, n_input, device=self.device)
        self.scales = [2 ** i for i in range(n_layers)]
        self.batch_first = batch_first
        self.nonlinearity = nonlinearity
        self.n_hidden = n_hidden
        self.depth = 0

    def forward(self, inputs) :
        if self.batch_first :
            inputs = inputs.transpose(0,1)

        largest_scale = self.scales[-1]
        hidden = [torch.randn(inputs.shape[1], self.n_hidden, device=self.device) for _ in range(len(self.scales))]
        p_inputs = self.pad_input(inputs, largest_scale)

        for i in range(len(inputs)) :
            if self.depth < len(self.scales) :
                target_scale = self.scales[self.depth]-1
                if i == target_scale:
                    self.depth += 1
            scaled_x_list = self.scaled_input(p_inputs[i:i+largest_scale, :, :], self.depth)
            hidden = self.apply_layers(scaled_x_list, hidden, self.depth)
            hidden = self.pad_hidden(hidden, self.depth)

        for i in range(len(hidden)) :
            hidden[i] = self.linear(hidden[i])

        return torch.stack(hidden)

    def scaled_input(self, x, depth) :
        scaled_x_list = []
        for i in range(depth) :
            scaled_x = x.unfold(0, self.scales[i], 1)
            scaled_x = scaled_x.mean(dim=3)[-1,:,:]
            scaled_x_list.append(scaled_x)
        return scaled_x_list

    def apply_layers(self, scaled_x_list, hidden, depth) :
        i = scaled_x_list[0]
        hidden_out = []
        for ind in range(depth) :
            s = scaled_x_list[ind]
            h = hidden[ind]
            cell = self.cells[ind]
            if self.nonlinearity == 'tanh':
                hidden_node = torch.tanh(cell.gate_i(i) + cell.gate_s(s) + cell.gate_h(h))
            elif self.nonlinearity == 'relu':
                hidden_node = torch.relu(cell.gate_i(i) + cell.gate_s(s) + cell.gate_h(h))
            hidden_out.append(hidden_node)
            i = hidden_node

        return hidden_out

    def pad_input(self, inputs, largest_scale):
        zeros = torch.zeros(largest_scale-1, inputs.shape[1], inputs.shape[2], device=self.device)
        padded_input = torch.cat((inputs, zeros))

        return padded_input

    def pad_hidden(self, hidden, depth):
        for _ in range(self.scales[-1]-depth-1) :
            randoms = torch.randn(hidden[0].shape, device=self.device)
            hidden.append(randoms)

        return hidden


class SRNNCell(nn.Module) :

    def __init__(self, input_size, hidden_size, scaled_size, device):
        super(SRNNCell, self).__init__()
        self.n_hidden = hidden_size
        self.gate_i = nn.Linear(input_size, hidden_size, device=device)
        self.gate_s = nn.Linear(scaled_size, hidden_size, device=device)
        self.gate_h = nn.Linear(hidden_size, hidden_size, device=device)