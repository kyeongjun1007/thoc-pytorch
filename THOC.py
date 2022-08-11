import torch
import torch.nn as nn
from sklearn.cluster import KMeans


class THOC(nn.Module):
    def __init__(self, n_input, n_hidden, n_layers, n_centroids, lambda_ = [0.1, 0,1], dropout=0, cell_type='GRU', batch_first=False):
        super().__init__()
        self.drnn = DRNN(n_input, n_hidden, n_layers, dropout, cell_type, batch_first)              # drnn 모델 생성
        self.n_layers = n_layers
        self.n_centroids = n_centroids                                                              # layer별 cluster center의 개수
        self.cluster_centers = [[[0]*n_hidden]*i for i in n_centroids]                              # cluster_centers를 layer별 n_centorids개 만큼의 n_hidden 차원의 0벡터로 초기화
        self.lambda_orth = lambda_[0]                                                               # threshold of loss_orth
        self.lambda_tss = lambda_[1]                                                                # threshold of loss_tss
        self.cos = nn.CosineSimilarity(dim=0)                                                       # cosine similarity layer
        self.mlp = nn.Linear(2*n_hidden, n_hidden)                                                  # MLP layer for concat function
        self.linear = nn.Linear(n_hidden, n_hidden)                                                 # linear layer for update function
        self.relu = nn.ReLU()                                                                       # relu layer for update function
             
    def forward(self, x, first = False):
        out, hidden = self.drnn(x)                                                                  # out = torch.tensor[[dim of X]*T]
        R = []
        if first == True :                                                                     # 첫 배치일때는 drnn의 노드값에 k-means를 적용하여 cluster centroids 초기화
            for i, n_clusters in enumerate(self.n_centroids) :
                k = n_clusters
                model = KMeans(n_clusters = k)
                model.fit(out[i].detach().numpy())
                self.cluster_centers[i] = model.cluster_centers_
        
        for layer in range(self.n_layers) :
            if (layer == 0) :
                f_bar = torch.stack([out[layer]])
            P = self.assign_prob(f_bar, self.cluster_centers[layer])
            R = self.calculate_R(P, R, layer)
            f_hat = self.update(f_bar, P)
            if (layer != (self.n_layer-1)) :
                f_bar = self.concat(f_hat,out[layer+1])
            
        KL = self.n_centroids[len(self.n_centroids)-1]
        
        anomaly_score = torch.matmul(torch.t(R),(1-self.cos(f_hat,self.cluster_centers[-1])))/KL            # sum(R*d)/K^L
        
        return anomaly_score
        
    def assign_prob(self, f_bar, centroids):
        P = []
        prob = []
        scores = []
        for i in range(f_bar.shape[0]):
            for t in range(f_bar.shape[1]) :
                for j in range(len(centroids)) :
                    scores.append(self.cos(f_bar[i,t], torch.tensor(centroids[j])).tolist())
                score_sum = sum(scores)
                scores = [i/score_sum for i in scores]
                prob.append(scores)
            P.append(prob)
        P = torch.stack(P)
        return P
    # P = [[[*,*,*,*]x6]xT]
    
    def update(self, f_bar, prob):
        f_hat = []
        for t in range(f_bar.shape[1]) :
            l = []
            for k in range(f_bar.shape[0]):
                y = self.linear(f_bar[k,t])
                y = self.relu(y)
                f_hat_list = [y*prob for prob in prob[t,k].tolist()]
                f_hat_list = torch.stack(f_hat_list)
                l.append(f_hat_list)
            l = torch.stack(l)
            l = torch.sum(l,0)
            f_hat.append(l)
        f_hat = torch.stack(f_hat)
        return f_hat
    #F_hat = [[[dim of X]*4]*T]
    
    def concat(self, f_hat, f):
        f_bar = []
        for k in range(f.shape[0]):
            in_f = torch.cat([f_hat, f[k]], dim=1)
            out_f = self.mlp(in_f)
            f_bar.append(out_f)
        f_bar = torch.stack(f_bar)
        return f_bar
    # f_bar = [[[dim of X]*T]*6]
    
    def calculate_R(self, P, R_, layer):                                                                       # 2차원 P를 받아서 1차원 R을 내보냄
        R = []
        if (layer==0):
            for t in range(P.shape[0]):
                R.append([(P[t,0,i]/sum(P[t,0])).tolist() for i in range(P.shape[2])])
        else :
            for t in range(P.shape[0]):
                k = []
                for i in range(P.shape[2]):
                    k.append(sum([P[t,c,i].tolist()*R_[t,c].tolist() for c in range(R_.shape[1])]))
                R.append([k[j]/sum(k) for j in range(len(k)-1)])
        R = torch.tensor(R)
        return R
    
##-------------------------------------------------DRNN---------------------------------------------------

import torch
import torch.nn as nn
class DRNN(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers, dropout=0, cell_type='GRU', batch_first=False):
        super(DRNN, self).__init__()

        self.dilations = [2 ** i for i in range(n_layers)]
        self.cell_type = cell_type
        self.batch_first = batch_first
        self.n_hidden = n_hidden

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
        layers = []
        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if hidden is None:
                inputs, _ = self.drnn_layer(cell, inputs, dilation)
            else:
                inputs, hidden[i] = self.drnn_layer(cell, inputs, dilation, hidden[i])

            outputs.append(inputs[-dilation:])
            layers.append(inputs.view(-1,self.n_hidden))

        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        return layers, outputs

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