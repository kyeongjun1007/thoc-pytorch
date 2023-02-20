import torch
import torch.nn as nn
from kmeans_pytorch import kmeans
from torch.nn.parameter import Parameter

use_cuda = torch.cuda.is_available()


class THOC(nn.Module):
    def __init__(self, n_input, n_hidden, n_layers, n_centroids, init_data, tau=1, tau_decay=2/3, dropout=0, cell_type='GRU',
                 batch_first=False):
        super().__init__()
        if use_cuda:
            self.device = "cuda:1"
        else:
            self.device = 'cpu'
        self.drnn = DRNN(n_input, n_hidden, n_layers, dropout, cell_type, batch_first)
        self.n_layers = n_layers  # layer 개수
        self.n_centroids = n_centroids  # layer별 cluster center의 개수
        self.hiddens = self.drnn(init_data)[0]  # 1st window의 drnn output (cluster centroids initializing에 사용)
        self.cluster_centers = Parameter(
            torch.stack([self.pad_tensor(kmeans(X=self.hiddens[i].flatten(0,1), device=self.device,
                                                num_clusters=n_clusters)[1], i) for i, n_clusters in
                         enumerate(self.n_centroids)]), requires_grad=True)
        self.cos = nn.CosineSimilarity(dim=3)  # cosine similarity layer
        self.cos_for_dist = nn.CosineSimilarity(dim=2)
        self.mlp = [nn.Linear(2 * n_hidden, n_hidden).to(self.device) for _ in range(n_layers)]  # MLP layer for concat function
        self.linear = nn.Linear(n_hidden, n_hidden)  # linear layer for update function
        self.relu = nn.ReLU()  # relu layer for update function
        self.self_superv = nn.Linear(n_hidden, n_input)
        self.tau = tau  # assign probability between f_bar & centroids
        self.tau_decay = tau_decay
        self.batch_first = batch_first

    def forward(self, x, i, epoch = None):

        # L : # of layers
        # T : length of input data
        # B : batch size
        # H : hidden size

        if epoch is None :
            pass
        elif (epoch != 0) and ((epoch % 5) == 0) and (i == 1):
            self._tau_decay()

        out, hidden = self.drnn(x)  # out = Tensor : (L, T, B, H)
        if self.batch_first:
            R = torch.ones(1, x.shape[1], x.shape[0], device=self.device)
        else:
            R = torch.ones(1, x.shape[0], x.shape[1], device=self.device)

        for layer in range(self.n_layers):
            if layer == 0:
                f_bar = torch.stack([out[layer]])                           # Tensor : (1, T, B, H);
            P = self.assign_prob(f_bar, self.unpad_tensor(self.cluster_centers[layer], layer))
            R = self.calculate_R(P, R)
            f_hat = self.update(f_bar, P)
            if layer != (self.n_layers - 1):
                f_bar = self.concat(f_hat, out[layer + 1], layer)

        cluster_L = self.unpad_tensor(self.cluster_centers[-1],self.n_layers-1)
        cosine_distance = []
        for i in range(len(cluster_L)) :
            cosine_similarity = self.cos_for_dist(f_hat[i], cluster_L[i])
            cosine_distance.append(torch.ones(cosine_similarity.shape, device=self.device)-cosine_similarity)
        cosine_distance = torch.stack(cosine_distance)

        anomaly_score = torch.mul(R, cosine_distance)

        diff_list = []

        for layer, C in enumerate(self.cluster_centers):
            unpaded_C = self.unpad_tensor(C, layer)
            CI = torch.matmul(torch.t(unpaded_C), unpaded_C) - torch.eye(unpaded_C.size(1)).to(self.device)
            diff_list.append((CI ** 2).mean())
        centroids_diff = torch.stack(diff_list)

        out_f = self.self_superv(out).transpose(1,2)

        return anomaly_score, centroids_diff, out_f

    def assign_prob(self, f_bar, centroids):
        P_list = []
        for i in range(len(centroids)) :
            cs = torch.exp(self.cos(f_bar, centroids[i])/self.tau)
            P_list.append(cs)
        P_tensor = torch.stack(P_list)
        P_sum = P_tensor.sum(0)
        P = torch.div(P_tensor, P_sum)
        return P

    # P = Tensor : (n_clusters[layer+1], n_clusters[layer], T, H)

    def update(self, f_bar, prob):
        T = f_bar.shape[1] # length of input data
        B = f_bar.shape[2] # batch size of input data
        H = f_bar.shape[3] # hidden size
        f_hat_linear = self.linear(f_bar)
        f_hat_relu = self.relu(f_hat_linear)
        f_hat_reshape = f_hat_relu.reshape(H,-1,T,B)
        f_hat_list = []
        for i in range(len(prob)) :
            f_hat_list.append(torch.mul(f_hat_reshape, prob[i]).reshape(-1,T,B,H))
        f_hats = torch.stack(f_hat_list)
        f_hat = torch.sum(f_hats, 1)
        return f_hat

    # f_hat = Tensor : (n_clusters[layer+1], T, B, H)

    def concat(self, f_hat, f, layer):
        concat_list = []
        for i in range(len(f_hat)) :
            concat = torch.cat([f_hat[i], f], dim=2)
            concat_list.append(self.mlp[layer](concat))
        f_bar = torch.stack(concat_list)
        return f_bar

    # f_bar = Tensor : (n_clusters[layer+1], T, B, H)

    def calculate_R(self, P, R_):
        R_hat = torch.mul(P, R_)
        R = R_hat.sum(1)
        return R

    # R = Tensor : (n_clusters[layer+1], T, B)

    def pad_tensor(self, short_tensor, layer):
        if layer == 0:
            paded_tensor = short_tensor
        else:
            n_centroid = self.n_centroids[layer]
            zero_tensor = torch.zeros(self.n_centroids[0] - n_centroid, short_tensor.shape[1], device=self.device)
            paded_tensor = torch.cat((short_tensor, zero_tensor))
        return paded_tensor

    def unpad_tensor(self, long_tensor, layer):
        if layer == 0:
            unpaded_tensor = long_tensor
        else:
            n_centroid = self.n_centroids[layer]
            unpaded_tensor = long_tensor[:n_centroid]
        return unpaded_tensor

    def _tau_decay(self):
        self.tau = self.tau * self.tau_decay

##-------------------------------------------------DRNN---------------------------------------------------


class DRNN(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers, dropout=0, cell_type='GRU', batch_first=False):
        super(DRNN, self).__init__()
        if use_cuda:
            self.device = "cuda:1"
        else:
            self.device = 'cpu'
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
                c = cell(n_input, n_hidden, dropout=dropout, device=self.device)
            else:
                c = cell(n_hidden, n_hidden, dropout=dropout, device=self.device)
            layers.append(c)
        self.cells = nn.Sequential(*layers)

    def forward(self, inputs, hidden=None):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)

        layers = torch.zeros(len(self.dilations), inputs.shape[0], inputs.shape[1], self.n_hidden, device=self.device)
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
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size,
                                                       hidden=hidden)

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
                                 inputs.size(2), device=self.device)

            inputs = torch.cat((inputs, zeros_))
        else:
            dilated_steps = n_steps // rate
            if use_cuda:
                inputs = inputs.to(self.device)

        return inputs, dilated_steps

    # inputs.shape = (max(len(inputs), dilation), n_input, batch_size)

    def _prepare_inputs(self, inputs, rate):
        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)
        return dilated_inputs

    def _prepare_hiddens(self, hiddens, rate):
        if (len(hiddens) >= rate):
            dilated_hiddens = torch.cat([hiddens[j::rate, :, :] for j in range(rate)], 1)
        else:
            dilated_hiddens = torch.cat([hiddens[j::rate, :, :] for j in range(len(hiddens))], 1)
            dilated_hiddens = torch.cat([dilated_hiddens, torch.zeros(1, rate - len(hiddens), self.n_hidden)], 1)
        return dilated_hiddens

    def init_hidden(self, batch_size, hidden_dim):
        hidden = torch.zeros(batch_size, hidden_dim, device=self.device)
        if self.cell_type == "LSTM":
            memory = torch.zeros(batch_size, hidden_dim, device=self.device)
            return (hidden, memory)
        else:
            return hidden