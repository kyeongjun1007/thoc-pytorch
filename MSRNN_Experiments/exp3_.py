from exp_tools.exp1_tools import VRNN_Copy, DRNN_Copy, MSRNN_Copy
from exp_tools.exp3_tools import sequence_generator
from torch.utils.tensorboard import SummaryWriter
import logging
from pathlib import Path
import torch.multiprocessing as mp
import time

import torch
import torch.nn as nn
from itertools import product


def check_debug():
    import sys

    eq = sys.gettrace() is None

    return not eq


T = 100
n_components = 4

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# logging config
path = "./log/"
if not Path(path).exists():
    Path(path).mkdir(parents=True)

if not check_debug():
    logging.basicConfig(level=logging.INFO, filename=path + f"/exp3_{n_components}.log", filemode="w", format="%(message)s")
    writer = SummaryWriter(f"./runs/exp3/{n_components}")
else:
    logging.basicConfig(level=logging.INFO, filename=path + f"/_debug_exp3_{n_components}.log", filemode="w",
                        format="%(message)s")
    writer = SummaryWriter("./runs/debug/exp3")

## data config
sequence, answer = sequence_generator(T, n_components)


def train(*cnm, **default_params):
    global writer
    global sequence
    global answer

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    cell_type, model_type = cnm

    ## model config
    n_input = 1
    n_output = len(answer)
    batch_first = True
    dropout = default_params['dropout']

    if model_type == 'SV':
        n_hidden = 2 ^ (default_params['layers'] - 1)
        n_layers = 1
    else:
        n_hidden = default_params['n_hidden']
        n_layers = default_params['layers']

    ## model choice
    if model_type in ['SV', 'MV']:
        model = VRNN_Copy(input_size=n_input, hidden_size=n_hidden, num_layers=n_layers,
                          cell_type=cell_type, dropout=dropout, out_size=n_output, batch_first=batch_first)
    elif model_type == 'D':
        model = DRNN_Copy(input_size=n_input, hidden_size=n_hidden, num_layers=n_layers,
                          cell_type=cell_type, dropout=dropout, out_size=n_output, batch_first=batch_first)
    elif model_type == 'MS':
        model = MSRNN_Copy(input_size=n_input, hidden_size=n_hidden, num_layers=n_layers,
                           cell_type=cell_type, dropout=dropout, out_size=n_output, batch_first=batch_first)
    else:
        raise ValueError("cell_type is not valid")

    model = model.to(device)
    sequence = sequence.to(device)
    answer = answer.to(device)

    ## train config
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=default_params['learning_rate'], alpha=default_params['alpha'])

    ## train
    iters = default_params['iters']

    model.train()
    total_loss = 0
    start_time = time.time()
    correct = 0
    counter = 0
    for iter in range(0, iters):

        optimizer.zero_grad()
        out = model(sequence.unsqueeze(2).contiguous())  # out: (batch, steps, output_size)

        loss = criterion(out.flatten(0,1), answer.flatten())
        pred = out.flatten(0,1).data.max(1, keepdim=True)[1]
        correct += pred.eq(answer.reshape(pred.shape)).cpu().sum()
        counter += out.flatten(0,1).size(0)
        if default_params['clip'] > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), default_params['clip'])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        writer.add_scalar(f'Loss/{model_type + cell_type}', loss.item(), iter)
        writer.add_scalars(f'Loss_all', {model_type + cell_type: loss.item()}, iter)

        # if iter > 0 and iter % default_params['log_interbal'] == 0:
        #     avg_loss = total_loss / default_params['log_interbal']
        #     elapsed = time.time() - start_time
        #     print('| Model {} | {:5d}/{:5d} batches | lr {:2.4f} | ms/batch {:5.2f} | '
        #           'loss {:5.8f} | accuracy {:5.4f}'.format(
        #         model_type+cell_type, iter, iters, default_params['learning_rate'], elapsed * 1000 / default_params['log_interbal'],
        #         avg_loss, 100. * correct / counter))
        #     start_time = time.time()
        #     total_loss = 0
        #     correct = 0
        #     counter = 0


def run(default_params, model_types, cell_types):
    num_proc = default_params['num_proc']

    async_result = mp.Queue()

    def __callback(value):
        async_result.put(value)

    pool = mp.Pool(num_proc)

    for cnm in [(c_type, m_type) for c_type, m_type in product(cell_types, model_types)]:
        pool.apply_async(train, args=cnm, kwds=default_params, callback=__callback)

    pool.close()
    pool.join()


if __name__ == '__main__':

    default_params = {
        # model params
        'n_hidden': 10,
        'layers': 9,

        # data params
        'T': T,
        'sequence_length': 10,
        'iters': 30000,

        # train params
        'learning_rate': 5e-04,
        'alpha': 0.9,
        'dropout': 0.0,
        'clip': 0.0,

        'log_interbal' : 10000,
        'num_proc': 4 if not check_debug() else 1}
    if not check_debug():
        model_types = ['SV', 'MV', 'D', 'MS']
        cell_types = ['RNN', 'LSTM', 'GRU']
    else :
        model_types = ['MS']
        cell_types = ['RNN', 'LSTM', 'GRU']

    run(default_params, model_types, cell_types)
