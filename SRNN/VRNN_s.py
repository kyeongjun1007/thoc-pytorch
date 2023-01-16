import torch
import torch.nn as nn


class VRNN (nn.RNN) :
    def __init__(self, n_input, n_hidden, n_layers, nonlinearity='tanh'):
        super(VRNN, self).__init__(n_input, n_hidden, n_layers, nonlinearity='tanh')
        self.linear = nn.Linear(n_hidden, n_input)

    def fit_dim(self, out):
        out_input_dim = self.linear(out)

        return out_input_dim

