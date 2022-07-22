import torch
import torch.nn as nn
from sklearn.cluster import KMeans

class THOC(nn.Module):
    def __init__(self, n_input, n_hidden, n_layers, n_centroids, dropout=0, cell_type='GRU', batch_first=False, first = False):
        super().__init__()
        self.first = first                                                                          # 첫 배치인가?
        self.drnn = DRNN(n_input, n_hidden, n_layers, dropout, cell_type, batch_first)              # drnn 모델 생성
        self.n_centroids = n_centroids                                                              # layer별 cluster center의 개수
        self.cluster_centers = [[[0]*n_hidden]*i for i in n_centroids]                              # cluster_centers를 layer별 n_centorids개 만큼의 n_hidden 차원의 0벡터로 초기화
             
    def forward(self, x):
        out, hidden = self.drnn(x)                                                                  '''n_layer개 만큼의 out이 나오도록 어떻게 쓰지(O)'''
        R = []
        if self.first == True :                                                                     # 첫 배치일때는 drnn의 노드값에 k-means를 적용하여 cluster centroids 초기화
            for i, n_clusters in enumerate(self.n_centroids) :
                k = n_clusters
                model = KMeans(n_clusters = k)
                model.fit(out)
                self.cluster_centers[i] = model.cluster_centers_
        
        for layer in range(n_layers) :
            if (layer == 0) :
                f_bar = out[layer]
            P = assign_prob(f_bar, self.cluster_centers)
            R = calculate_R(P)
            f_hat = update(f_bar, P)
            if (layer != (n_layer-1)) :
                f_bar = concat(f_hat,out[layer+1])
            
        KL = self.n_centroids[len(self.n_centroids)-1]
        
        loss_thoc = torch.matmul(torch.t(R),cos_dist(f_hat,self.cluster_centers[-1]))/KL            # sum(R*d)/K^L
        
        co = torch.matmul(torch.t(self.cluster_centers),self.cluster_centers) -
             torch.eye(self.cluster_centers.size(0))                                                # co = t(C)*C-I
        co = np.linalg.norm(co, ord='fro')                                                          # co = frobenius norm of co
        loss_orth = (co*co)/KL                                                                      # co^2/the num of last layer centroids
        
        loss_tss = 
        
    def assign_prob(self, f_bar, centroids):
        return prob
    
    def update(self, f_bar, prob):
        return f_hat
    
    def concat(self, f_hat, f):
        return f_bar
    
    def calculate_R(self, P):
        return R
    
    def cos_dist(self, f_L,c_L):
        return d
# git commit 왜 안되지
##-------------------------------------------------DRNN---------------------------------------------------
'''lyaer별로 out값을 내보내도록 수정해야함...! (h_n 이용?)'''
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