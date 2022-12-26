import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class WRNN(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers, batch_first = True, nonlinearity = 'tanh'):
        super().__init__()
        self.n_hidden = n_hidden
        self.scales =[2 ** i for i in range(n_layers)]
        self.batch_first = batch_first
        self.nonlinearity = nonlinearity
        self.weight_s = [torch.randn(n_input, n_hidden) for _ in range(n_layers)]
        self.weight_h = [torch.randn(n_hidden, n_hidden) for _ in range(n_layers)]
        self.weight_i = [torch.randn(n_input, n_hidden)] + [torch.randn(n_hidden, n_hidden) for _ in range(n_layers - 1)]
        self.bias_s = [torch.randn(n_hidden) for _ in range(n_layers)]
        self.bias_h = [torch.randn(n_hidden) for _ in range(n_layers)]
        self.bias_i = [torch.randn(n_hidden) for _ in range(n_layers)]

    def forward(self, x):
        out_all = []
        for layer, scale in enumerate(self.scales) :
            scaled_x = self.scaled_input(x, scale)
            out = self.wrnn_layer(scaled_x, out, layer)
            out_all.append(out)

        return out_all

    def scaled_input(self, x, scale):
        scaled_x = x.unfold(0,scale,1)
        scaled_x = torch.mean(scaled_x, 3)
        return scaled_x

    def wrnn_layer(self, scaled_x, input, layer) :
        # 인자 세개 차원 맞는지 확인하는 코드 추가
        # ReLU 추가
        # sclaed_x of shape (batch, input size),
        # input of shape (batch, input size),
        # hidden of shape (batch, hidden size)
        out = []
        hidden_t = torch.zeros(scaled_x.shape[1], self.n_hidden)
        for i in range(len(scaled_x)) :
            if self.nonlinearity == 'tanh' :
                hidden_t = torch.tanh(torch.matmul(input[self.scales[layer]-1+i], self.weight_i[layer]) +
                                      torch.matmul(hidden_t, self.weight_h[layer]) +
                                      torch.matmul(scaled_x[i], self.weight_s[layer]))
            elif self.nonlinearity == 'relu' :
                hidden_t = torch.relu(torch.matmul(input[self.scales[layer] - 1 + i], self.weight_i[layer]) +
                                      torch.matmul(hidden_t, self.weight_h[layer]) +
                                      torch.matmul(scaled_x[i], self.weight_s[layer]))
            else :
                raise
            out.append(hidden_t)

        out = torch.stack(out)

        return out
