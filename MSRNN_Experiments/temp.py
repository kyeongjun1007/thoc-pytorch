import torch
from exp_tools.exp1_tools import VRNN_Copy, DRNN_Copy, MSRNN_Copy, data_generator

n_input = 1
n_hidden = 10
n_output = 10
n_layers = 10
batch_first = True
dropout = 0.0
cell_type = 'LSTM'


model = VRNN_Copy(input_size=n_input, hidden_size=n_hidden, num_layers=n_layers,
                          cell_type=cell_type, dropout=dropout, out_size=n_output, batch_first=batch_first)

torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)
c_type = 'LSTM'
m_type = 'SV'
'| Model {} | {:5d}/{:5d} batches | lr {:2.4f} | ms/batch {:5.2f} | loss {:5.8f} | accuracy {:5.4f}'.format(c_type+m_type, 50, 100, 0.001, 5.54654,27.189649846, 65.16468)