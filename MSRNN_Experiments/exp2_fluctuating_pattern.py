from itertools import product
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import logging
import torch.multiprocessing as mp

from Models import VRNN, VLSTM, VGRU, MSRNN, DRNN
from exp_tools.exp2_tools import DataGenerator, SlidingWindow
from torch.utils.data import DataLoader

# data config
amplitude = [1.0, 3.0, 5.0, 7.0]
frequency = [1/2, 1/4, 1/8, 1/16]
phase = [0.0]*4
t = 20

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


def experiment(cnm, scenario_num) :

    global writer
    global data_generator

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    cell_type, model_type = cnm

    ## model config
    n_input = 1
    n_output = 10
    batch_first = True

    if cell_type == 'SV':
        n_hidden = 64
        n_layers = 1
    else:
        n_hidden = 8
        n_layers = 4

    ## model choice
    if cell_type in ['SV', 'MV']:
        if model_type == 'RNN':
            model = VRNN(n_input, n_hidden, n_output, n_layers, batch_first).to(device)
        elif model_type == 'LSTM':
            model = VLSTM(n_input, n_hidden, n_output, n_layers, batch_first).to(device)
        elif model_type == 'GRU':
            model = VGRU(n_input, n_hidden, n_output, n_layers, batch_first).to(device)
    elif cell_type == 'D':
        model = DRNN(n_input, n_hidden, n_output, n_layers, cell_type=model_type, batch_first=batch_first)
    elif cell_type == 'MS':
        model = MSRNN(n_input, n_hidden, n_output, n_layers, cell_type=model_type, batch_first=batch_first)
    else:
        raise ValueError("cell_type is not valid")

    ## training config
    num_epochs = 1
    learning_rate = 0.001
    batch_size = 1
    window_size = 16

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    data = data_generator.get_data(scenario_num)

    model = model.to(device)
    data = data.to(device)

    dataset = SlidingWindow(data, window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # training
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs, _ = model(x)
            loss = criterion(outputs, y[-len(outputs):])
            loss.backward()
            optimizer.step()

            writer.add_scalar(f'Loss/{cell_type + model_type}', loss.item(), epoch)
            writer.add_scalars(f'Loss_all', {cell_type + model_type: loss.item()}, epoch)


def run(args) :
    cell_types, model_types, scenario_nums, num_proc = args

    async_result = mp.Queue()

    def __callback(value) :
        async_result.put(value)

    pool = mp.Pool(num_proc)
    cnms = [(c_type, m_type) for c_type, m_type in product(cell_types, model_types)]

    for cnm, scenario_num in product(cnms, scenario_nums) :
        pool.apply_async(experiment, args=(cnm, scenario_num), callback=__callback)

    pool.close()
    pool.join()


if __name__ == '__main__' :

    # basic config
    cell_types = ['SV', 'MV', 'D', 'MS']
    model_types = ['RNN', 'LSTM', 'GRU']
    scenario_nums = ['1', '2', '3', '4', '5']
    num_proc = 3

    args = (cell_types, model_types, scenario_nums, num_proc)

    run(args)