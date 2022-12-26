import torch
import torch.nn as nn
from kmeans_pytorch import kmeans
from torch.nn.parameter import Parameter

use_cuda = torch.cuda.is_available()

class THOC(nn.Module):
    def __init__(self, n_input, n_hidden, n_layers, n_centroids, x, tau=1, dropout=0, cell_type='GRU', batch_first=False):
        super().__init__()
        if use_cuda :
            self.drnn = DRNN(n_input, n_hidden, n_layers, dropout, cell_type, batch_first).cuda()                           # drnn 모델 생성
        else :
            self.drnn = DRNN(n_input, n_hidden, n_layers, dropout, cell_type, batch_first)
        self.n_layers = n_layers                                                                                            # layer 개수
        self.n_centroids = n_centroids                                                                                      # layer별 cluster center의 개수
        self.hiddens = self.drnn(x)[1]                                                                                          # 1st window의 drnn output (cluster centroids initializing에 사용)
        if use_cuda :
            self.cluster_centers = Parameter(torch.stack([self.pad_tensor(kmeans(X=self.hiddens[i].view(-1, n_hidden), device='cuda',
                                             num_clusters=n_clusters)[1], i) for i, n_clusters in enumerate(self.n_centroids)]), requires_grad=True)
        else :
            self.cluster_centers = Parameter(torch.stack([self.pad_tensor(kmeans(X=self.hiddens[i].view(-1, n_hidden),
                                             num_clusters=n_clusters)[1], i) for i, n_clusters in enumerate(self.n_centroids)]), requires_grad=True)
        self.cos = nn.CosineSimilarity(dim=0)                                                                               # cosine similarity layer
        self.mlp = nn.Linear(2*n_hidden, n_hidden)                                                                          # MLP layer for concat function
        self.linear = nn.Linear(n_hidden, n_hidden)                                                                         # linear layer for update function
        self.relu = nn.ReLU()                                                                                               # relu layer for update function
        self.self_superv = nn.Linear(n_hidden, n_input)
        self.tau = tau                                                                                                      # assign probability between f_bar & centroids
        self.batch_first = batch_first

        if use_cuda :
            self.device = "cuda:0"
        else :
            self.device = 'cpu'

    def forward(self, x):
        out, hidden = self.drnn(x)                                                                                          # out = torch.tensor[[[[dim of X]*batch_size]*T]*layer]
        if self.batch_first :
            R = torch.ones(x.shape[0], x.shape[1], 1)
        else :
            R = torch.ones(x.shape[1], x.shape[0], 1)
        # if use_cuda :
        #     R = R.cuda()

        for layer in range(self.n_layers) :
            if layer == 0:
                f_bar = torch.stack([out[layer]])                                                                           # f_bar = torch.tensor[[[[dim_of_X]*batch_size]*T]*1]
            P = self.assign_prob(f_bar, self.unpad_tensor(self.cluster_centers[layer], layer))
            R = self.calculate_R(P, R)
            f_hat = self.update(f_bar, P)
            if layer != (self.n_layers - 1):
                f_bar = self.concat(f_hat,out[layer+1])

        anomaly_score = []
        for b in range(f_hat.shape[0]) :
            for t in range(f_hat.shape[1]) :
                anomaly = 0
                for c in range(self.n_centroids[-1]) :
                    for f in range(f_hat.shape[2]) :
                        anomaly += R[b,t,f]*(1-self.cos(f_hat[b,t,f], self.cluster_centers[-1][c]))
                anomaly_score.append(anomaly)

        diff_list = []
        for layer, C in enumerate(self.cluster_centers):
            unpaded_C = self.unpad_tensor(C,layer)
            CI = torch.matmul(torch.t(unpaded_C), unpaded_C) - torch.eye(unpaded_C.size(1)).to(self.device)
            diff_list.append((CI**2).mean())
        centroids_diff = torch.stack(diff_list)

        out_anomaly_score = torch.stack(anomaly_score)
        out_centroids_diff = centroids_diff
        out_f = self.self_superv(out)

        return out_anomaly_score, out_centroids_diff, out_f

    def assign_prob(self, f_bar, centroids):
        B_assign = []
        for b in range(f_bar.shape[2]) :
            P = []
            for i in range(f_bar.shape[0]):
                prob = []
                for t in range(f_bar.shape[1]) :
                    scores = []
                    for j in range(len(centroids)) :
                        scores.append(torch.exp(self.cos(f_bar[i,t,b], centroids[j])/self.tau))
                    score_sum = sum(scores)
                    scores_p = [score/score_sum for score in scores]
                    prob.append(scores_p)
                P.append(prob)
            B_assign.append(P)
        P_batch = torch.tensor(B_assign)
        return P_batch
    # P = torch.tensor([[[[n_centroids[layer+1]]*T]*n_centroids[layer]]*batch_size])

    def update(self, f_bar, prob):
        B_update = []
        for b in range(f_bar.shape[2]) :
            f_hat = []
            for t in range(f_bar.shape[1]) :
                l = []
                for k in range(f_bar.shape[0]):
                    y = self.linear(f_bar[k,t,b])
                    y = self.relu(y)
                    f_hat_list = [y*p for p in prob[b,k,t]]
                    f_hat_list = torch.stack(f_hat_list)
                    l.append(f_hat_list)
                l = torch.stack(l)
                l = torch.sum(l,0)
                f_hat.append(l)
            f_hat = torch.stack(f_hat)
            B_update.append(f_hat)
        f_hat_batch = torch.stack(B_update)
        return f_hat_batch
    # f_hat = torch.tensor([[[[dim of X]*n_centroids[layer+1]]*T]*batch_size])

    def concat(self, f_hat, f):
        f_bar = []
        for k in range(f_hat.shape[2]) :
            B_concat = []
            for b in range(f_hat.shape[0]):
                in_f = torch.cat([f_hat[b,:,k,:], f[:,b,:]], dim=1)
                out_f = self.mlp(in_f)
                B_concat.append(out_f)
            B_concat = torch.stack(B_concat).view(f_hat.shape[1],-1,f_hat.shape[3])
            f_bar.append(B_concat)
        f_bar_batch = torch.stack(f_bar)
        return f_bar_batch
    # f_bar = torch.tensor([[[[dim of X]*batch_size]*T]*n_centroids[layer+1]])

    def calculate_R(self, P, R_):
        B_calculate_R = []
        for b in range(P.shape[0]) :
            R_list = []
            for t in range(P.shape[2]) :
                k = []
                for i in range(P.shape[3]) :
                    k.append(sum([P[b, c, t, i].tolist() * R_[b, t, c].tolist() for c in range(R_.shape[2])]))
                r_sum = sum(k)
                k_exp = [r/r_sum for r in torch.exp(torch.tensor(k))]
                R_list.append(k_exp)
            B_calculate_R.append(R_list)
        B_calculate_R = torch.tensor(B_calculate_R)
        return B_calculate_R
    # R = [[[# of next layer cluster_centers]*T]*batch_size]

    def pad_tensor(self, short_tensor, layer):
        if layer == 0 :
            paded_tensor = short_tensor
        else :
            n_centroid = self.n_centroids[layer]
            zero_tensor = torch.zeros(self.n_centroids[0]-n_centroid, short_tensor.shape[1])
            paded_tensor = torch.cat((short_tensor, zero_tensor))
        return paded_tensor

    def unpad_tensor(self, long_tensor, layer):
        if layer == 0 :
            unpaded_tensor = long_tensor
        else :
            n_centroid = self.n_centroids[layer]
            unpaded_tensor = long_tensor[:n_centroid]
        return unpaded_tensor


##-------------------------------------------------DRNN---------------------------------------------------

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

        layers = torch.zeros(len(self.dilations), inputs.shape[0], inputs.shape[1], self.n_hidden).cuda()
        hiddens = []

        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if hidden is None:
                inputs, k = self.drnn_layer(cell, inputs, dilation)
            else:
                inputs, hidden[i] = self.drnn_layer(cell, inputs, dilation, hidden[i])
                k = hidden[i]

            layers[i] = inputs

            hiddens.append(k)

        return layers, hiddens

    def drnn_layer(self, cell, inputs, rate, hidden=None):
        n_steps = len(inputs)
        batch_size = inputs[0].size(0)
        hidden_size = cell.hidden_size

        inputs, _ = self._pad_inputs(inputs, n_steps, rate)
        dilated_inputs = self._prepare_inputs(inputs, rate)

        if hidden is None:
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)
        else:
            hidden = self._prepare_hiddens(hidden, rate)
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
            if use_cuda:
                inputs = inputs.cuda()

        return inputs, dilated_steps
    # inputs.shape = (max(len(inputs), dilation), n_input, batch_size)

    def _prepare_inputs(self, inputs, rate):
        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)
        return dilated_inputs

    def _prepare_hiddens(self, hiddens, rate):
        if (len(hiddens)>=rate) :
            dilated_hiddens = torch.cat([hiddens[j::rate, :, :] for j in range(rate)], 1)
        else :
            dilated_hiddens = torch.cat([hiddens[j::rate, :, :] for j in range(len(hiddens))], 1)
            dilated_hiddens = torch.cat([dilated_hiddens,torch.zeros(1,rate-len(hiddens),self.n_hidden)],1)
        return dilated_hiddens

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
