from exp_tools.exp4_tools import MSRNN_Copy, get_data_loader
from torch.utils.tensorboard import SummaryWriter
import logging
from pathlib import Path
import torch.multiprocessing as mp

import torch
import torch.nn as nn
from itertools import product
from torch.utils.tensorboard.summary import hparams


def check_debug():
    import sys

    eq = sys.gettrace() is None

    return not eq


permute = True

# logging config
path = "./log/"
if not Path(path).exists():
    Path(path).mkdir(parents=True)

if not check_debug():
    logging.basicConfig(level=logging.INFO, filename=path + f"/exp4_permute{permute}.log", filemode="w", format="%(message)s")
    writer = SummaryWriter(f"./runs/exp4/permute{permute}")
else:
    logging.basicConfig(level=logging.INFO, filename=path + f"/_debug_exp4_permute{permute}.log", filemode="w",
                        format="%(message)s")
    writer = SummaryWriter(f"./runs/debug/exp4/permute{permute}")


def train(*clr, **default_params):
    global writer

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    cell_type, lr, decay = clr

    ## model config
    n_input = 1
    n_output = 10
    batch_first = True
    dropout = default_params['dropout']

    n_hidden = default_params['n_hidden']
    n_layers = default_params['layers']

    model = MSRNN_Copy(input_size=n_input, hidden_size=n_hidden, num_layers=n_layers,
                       cell_type=cell_type, dropout=dropout, out_size=n_output, batch_first=batch_first)

    model = model.to(device)

    ## train config
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

    ## train
    train_dataloader, valid_dataloader = get_data_loader(default_params['batch_size'])
    for epoch in range(default_params['n_epochs']):
        model.train()
        for i, (x, y) in enumerate(train_dataloader):

            optimizer.zero_grad()
            x = x.to(device).float()
            y = y.to(device).long()

            out = model(x.unsqueeze(2))  # out: (batch, steps, output_size)

            loss = criterion(out[:, -1, :], y.flatten())
            if default_params['clip'] > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(), default_params['clip'])
            loss.backward()
            optimizer.step()

            # writer.add_scalar(f'Loss/{lr}_{alpha}', loss.item(), i)
            # writer.add_scalars(f'Loss_all', {f'{lr}_{alpha}': loss.item()}, i)

            if check_debug() :
                if i == 10 :
                    break

        total_loss = 0
        correct = 0
        counter = 0
        model.eval()
        for i, (x, y) in enumerate(valid_dataloader):

            x = x.to(device).float()
            y = y.to(device).long()

            out = model(x.unsqueeze(2))  # out: (batch, steps, output_size)

            loss = criterion(out[:, -1, :], y.flatten())
            pred = out[:, -1, :].data.max(1, keepdim=True)[1]
            correct += pred.eq(y.reshape(pred.shape)).cpu().sum()
            counter += out[:, -1, :].size(0)
            total_loss += loss.item()

            if check_debug() :
                if i == 10 :
                    break

        # param_dict = {
        #     'lr': lr,
        #     'alpha': alpha,
        # }
        #
        # result = {
        #     'loss': round(total_loss / len(valid_dataloader), 5),
        #     'correct' : correct,
        #     'counter' : counter,
        #     'acc': correct / counter
        # }
        #
        # writer.add_hparams(param_dict, result)
        writer.add_scalar(f'Loss/{cell_type}_{lr}_{decay}', total_loss/len(valid_dataloader), epoch)
        writer.add_scalar(f'Accuracy/{cell_type}_{lr}_{decay}', correct/counter, epoch)
        writer.add_scalars(f'Loss_all', {f'{cell_type}_{lr}_{decay}': total_loss/len(valid_dataloader)}, epoch)
        writer.add_scalars(f'Accuracy_all', {f'{cell_type}_{lr}_{decay}': correct/counter}, epoch)


def run(default_params, cell_types, learning_rates, decays):
    num_proc = default_params['num_proc']

    async_result = mp.Queue()

    def __callback(value):
        async_result.put(value)

    pool = mp.Pool(num_proc)

    for clr in [(c_type, lr, decay) for c_type, lr, decay in product(cell_types, learning_rates, decays)]:
        pool.apply_async(train, args=clr, kwds=default_params, callback=__callback)

    pool.close()
    pool.join()


if __name__ == '__main__':

    default_params = {
        # model params
        'n_hidden': 20,
        'layers': 9,

        'permute' : permute,

        # train params
        'n_epochs': 50,
        'batch_size' : 50,
        'dropout': 0.0,
        'clip': 0.0,

        'num_proc': 2 if not check_debug() else 1}

    if not check_debug():
        cell_types = ['RNN', 'LSTM', 'GRU']
    else :
        cell_types = ['RNN']

    learning_rates= [1e-03, 1e-04, 1e-05]
    decays = [0.3, 0.5, 0.7]

    run(default_params, cell_types, learning_rates, decays)
