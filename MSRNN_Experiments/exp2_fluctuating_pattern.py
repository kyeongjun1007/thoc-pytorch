from itertools import product
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import logging
import torch.multiprocessing as mp

from exp_tools.exp1_tools import VRNN_Copy, DRNN_Copy, MSRNN_Copy
from exp_tools.exp2_tools import DataGenerator, SlidingWindow
from torch.utils.data import DataLoader

# data config
amplitude = [1.0, 3.0, 5.0, 7.0]
frequency = [1/2, 1/4, 1/8, 1/16]
phase = [0.0]*4
seq_length = 40
t = torch.linspace(0, int(seq_length//2), seq_length+1)[:seq_length]

data_generator = DataGenerator(amplitude, frequency, phase, t)


def check_debug():
    import sys

    eq = sys.gettrace() is None

    return not eq


# logging config
path = "./log/"
if not Path(path).exists():
    Path(path).mkdir(parents=True)

if not check_debug():
    logging.basicConfig(level=logging.INFO, filename=path + "/exp2.log", filemode="w", format="%(message)s")
    writer = SummaryWriter("./runs/exp2")
else:
    logging.basicConfig(level=logging.INFO, filename=path + "/_debug_exp2.log", filemode="w",
                        format="%(message)s")
    writer = SummaryWriter(f"./runs/debug/exp2")


def experiment(*args, **default_params) :
    cnm, scenario_num = args

    global writer
    global data_generator

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    cell_type, model_type = cnm

    ## model config
    n_input = 1
    n_output = 1
    batch_first = True
    dropout = 0.0

    if model_type == 'SV':
        n_hidden = 2^(default_params['n_layers']-1)
        n_layers = 1
    else:
        n_hidden = default_params['n_hidden']
        n_layers = default_params['n_layers']

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

    ## training config
    num_epochs = default_params['num_epochs']
    learning_rate = default_params['learning_rate']
    batch_size = 1
    window_size = default_params['window_size']

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    data = data_generator.get_data(scenario_num)
    data = data.unsqueeze(1)

    model = model.to(device)
    data = data.to(device)

    dataset = SlidingWindow(data, window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # training
    for epoch in range(num_epochs):
        loss_sum = []
        for i, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y[:,-outputs.shape[1]:])
            loss.backward()
            optimizer.step()

            loss_sum.append(loss.item())

            # writer.add_scalars('Loss/scenario.'+scenario_num, {model_type+cell_type : loss.item()}, epoch*len(dataloader)+i)

        if check_debug() :
            continue
        loss_mean = sum(loss_sum)/len(loss_sum)
        writer.add_scalars('Loss/scenario.'+scenario_num, {model_type+cell_type : loss_mean}, epoch)


def run(args) :
    default_params, cell_types, model_types, scenario_nums, num_proc = args

    async_result = mp.Queue()

    def __callback(value) :
        async_result.put(value)

    pool = mp.Pool(num_proc)
    cnms = [(c_type, m_type) for c_type, m_type in product(cell_types, model_types)]

    for cnm, scenario_num in product(cnms, scenario_nums) :
        pool.apply_async(experiment, args=(cnm, scenario_num), kwds=default_params, callback=__callback)

    pool.close()
    pool.join()


if __name__ == '__main__' :

    default_params = {
        'n_hidden' : 8,
        'n_layers' : 4,
        'num_epochs' : 30,
        'learning_rate' : 0.001,
        'window_size' : 16,
    }

    # basic config
    model_types = ['SV', 'MV', 'D', 'MS']
    cell_types = ['RNN', 'LSTM', 'GRU']
    scenario_nums = ['1','2','3','4','5']
    num_proc = 4

    args = (default_params, cell_types, model_types, scenario_nums, num_proc)

    run(args)