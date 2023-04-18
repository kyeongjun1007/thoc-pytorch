from exp_tools.exp1_tools import VRNN_Copy, DRNN_Copy, MSRNN_Copy, data_generator
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


T = 500
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# logging config
path = "./log/"
if not Path(path).exists():
    Path(path).mkdir(parents=True)

if not check_debug():
    logging.basicConfig(level=logging.INFO, filename=path + f"/exp1_{T}.log", filemode="w", format="%(message)s")
    writer = SummaryWriter(f"./runs/exp1/{T}")
else:
    logging.basicConfig(level=logging.INFO, filename=path + f"/_debug_exp1_{T}.log", filemode="w",
                        format="%(message)s")
    writer = SummaryWriter("./runs/debug/exp1")


def train(*cnm, **default_params):
    global writer

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    cell_type, model_type = cnm

    ## data config
    train_x, train_y = data_generator(default_params['T'], default_params['sequence_length'], default_params['iters'])
    # 30000, 520

    ## model config
    n_input = 1
    n_output = 10
    batch_first = True
    dropout = default_params['dropout']

    if model_type == 'SV':
        n_hidden = 2 ^ (default_params['layers'] - 1)
        n_layers = 1
    else:
        n_hidden = default_params['n_hidden']
        n_layers = default_params['layers']
    n_classes = 10


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
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    ## train config
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=default_params['learning_rate'], alpha=default_params['alpha'])

    ## train
    iters = default_params['iters']
    batch_size = default_params['batch_size']

    model.train()
    total_loss = 0
    start_time = time.time()
    correct = 0
    counter = 0
    for batch_idx, batch in enumerate(range(0, iters, batch_size)):
        start_ind = batch
        end_ind = start_ind + batch_size

        x = train_x[start_ind:end_ind]  # (batch, steps)
        y = train_y[start_ind:end_ind]  # (batch, steps)

        optimizer.zero_grad()
        out = model(x.unsqueeze(2).contiguous())  # out: (batch, steps, output_size)

        out = out[:, -default_params['sequence_length']:]
        y = y[:, -default_params['sequence_length']:]

        loss = criterion(out.flatten(0,1), y.flatten())
        pred = out.flatten(0,1).data.max(1, keepdim=True)[1]
        correct += pred.eq(y.reshape(pred.shape)).cpu().sum()
        counter += out.flatten(0,1).size(0)
        if default_params['clip'] > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), default_params['clip'])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        writer.add_scalar(f'Loss/{model_type + cell_type}', loss.item(), batch_idx)
        writer.add_scalars(f'Loss_all', {model_type + cell_type: loss.item()}, batch_idx)

        if batch_idx > 0 and batch_idx % default_params['log_interbal'] == 0:
            avg_loss = total_loss / default_params['log_interbal']
            elapsed = time.time() - start_time
            print('| Model {} | {:5d}/{:5d} batches | lr {:2.4f} | ms/batch {:5.2f} | '
                  'loss {:5.8f} | accuracy {:5.4f}'.format(
                model_type+cell_type, batch_idx, iters // batch_size + 1, default_params['learning_rate'], elapsed * 1000 / default_params['log_interbal'],
                avg_loss, 100. * correct / counter))
            start_time = time.time()
            total_loss = 0
            correct = 0
            counter = 0


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
        'batch_size': 128,
        'learning_rate': 0.001,
        'alpha': 0.9,
        'dropout': 0.0,
        'clip': 0.0,

        'log_interbal' : 50,
        'num_proc': 4 if not check_debug() else 1}
    if not check_debug():
        model_types = ['SV', 'MV', 'D']
        cell_types = ['RNN', 'LSTM', 'GRU']
    else :
        model_types = ['MS']
        cell_types = ['RNN', 'LSTM', 'GRU']

    run(default_params, model_types, cell_types)
