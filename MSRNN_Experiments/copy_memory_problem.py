import torch
import torch.nn as nn
from Models import VRNN, VLSTM, VGRU, MSRNN, DRNN
from torch.utils.tensorboard import SummaryWriter
import logging
from pathlib import Path


# Config
path = "./log/copy_memory_problem/500"
if not Path(path).exists():
    Path(path).mkdir(parents=True)

logging.basicConfig(level=logging.INFO, filename=path + "/results.log", filemode="w", format="%(message)s")
writer = SummaryWriter("./runs/copy_memory_problem/500")

# Models init
n_input = 10
n_hidden = 10
n_out = 10
n_layers = 9
dropout = 0.0
batch_first = True

## Multi-Layer models init

mvrnn = VRNN(n_input, n_hidden, n_out, n_layers, dropout=dropout, batch_first=batch_first).to('cuda:0' if torch.cuda.is_available() else 'cpu')
mvlstm = VLSTM(n_input, n_hidden, n_out, n_layers, dropout=dropout, batch_first=batch_first).to('cuda:0' if torch.cuda.is_available() else 'cpu')
mvgru = VGRU(n_input, n_hidden, n_out, n_layers, dropout=dropout, batch_first=batch_first).to('cuda:0' if torch.cuda.is_available() else 'cpu')

## Dilated models init

drnn = DRNN(n_input, n_hidden, n_out, n_layers, dropout=dropout, cell_type='RNN', batch_first=batch_first)
dlstm = DRNN(n_input, n_hidden, n_out, n_layers, dropout=dropout, cell_type='LSTM', batch_first=batch_first)
dgru = DRNN(n_input, n_hidden, n_out, n_layers, dropout=dropout, cell_type='GRU', batch_first=batch_first)

## Multi-scale models init

msrnn = MSRNN(n_input, n_hidden, n_out, n_layers, dropout=dropout, cell_type='RNN', batch_first=batch_first)
mslstm = MSRNN(n_input, n_hidden, n_out, n_layers, dropout=dropout, cell_type='LSTM', batch_first=batch_first)
msgru = MSRNN(n_input, n_hidden, n_out, n_layers, dropout=dropout, cell_type='GRU', batch_first=batch_first)

## Single-Layer Vanilla models init
n_hidden = 256
n_layers = 1

svrnn = VRNN(n_input, n_hidden, n_out, n_layers, dropout=dropout, batch_first=batch_first).to('cuda:0' if torch.cuda.is_available() else 'cpu')
svlstm = VLSTM(n_input, n_hidden, n_out, n_layers, dropout=dropout, batch_first=batch_first).to('cuda:0' if torch.cuda.is_available() else 'cpu')
svgru = VGRU(n_input, n_hidden, n_out, n_layers, dropout=dropout, batch_first=batch_first).to('cuda:0' if torch.cuda.is_available() else 'cpu')

# Data generation
T = 500
memory_lenght = n_out


def generate_data(T, memory_length) :
    return torch.cat((torch.randint(0, 8, (memory_length,)), torch.ones(T-1)*8, torch.ones(memory_length+1)*9))


criterion = nn.CrossEntropyLoss()
svrnn_optimizer = torch.optim.RMSprop(svrnn.parameters(), lr=0.001, alpha=0.9)
svlstm_optimizer = torch.optim.RMSprop(svlstm.parameters(), lr=0.001, alpha=0.9)
svgru_optimizer = torch.optim.RMSprop(svgru.parameters(), lr=0.001, alpha=0.9)
mvrnn_optimizer = torch.optim.RMSprop(mvrnn.parameters(), lr=0.001, alpha=0.9)
mvlstm_optimizer = torch.optim.RMSprop(mvlstm.parameters(), lr=0.001, alpha=0.9)
mvgru_optimizer = torch.optim.RMSprop(mvgru.parameters(), lr=0.001, alpha=0.9)
drnn_optimizer = torch.optim.RMSprop(drnn.parameters(), lr=0.001, alpha=0.9)
dlstm_optimizer = torch.optim.RMSprop(dlstm.parameters(), lr=0.001, alpha=0.9)
dgru_optimizer = torch.optim.RMSprop(dgru.parameters(), lr=0.001, alpha=0.9)
msrnn_optimizer = torch.optim.RMSprop(msrnn.parameters(), lr=0.001, alpha=0.9)
mslstm_optimizer = torch.optim.RMSprop(mslstm.parameters(), lr=0.001, alpha=0.9)
msgru_optimizer = torch.optim.RMSprop(msgru.parameters(), lr=0.001, alpha=0.9)

sequence = generate_data(T, memory_lenght)
one_hot_sequence = torch.nn.functional.one_hot(sequence.type(torch.int64))
one_hot_sequence = one_hot_sequence.unsqueeze(0).float()
sequence = sequence.unsqueeze(0).unsqueeze(2).float()
target = sequence[:, :memory_lenght, :].squeeze().long().to(device='cuda:0' if torch.cuda.is_available() else 'cpu')
sequence = one_hot_sequence.to(device='cuda:0' if torch.cuda.is_available() else 'cpu')


for i in range(30000) :

    logging.info(f'Iteration: {i}')

    # Single-Layer Vanilla models
    svrnn_output_, _ = svrnn(sequence)
    svrnn_output = svrnn.fit_dim(svrnn_output_)
    svrnn_loss = criterion(svrnn_output.squeeze(0)[-memory_lenght:], target)

    svrnn_optimizer.zero_grad()
    svrnn_loss.backward()
    svrnn_optimizer.step()
    # svrnn_scheduler.step()

    logging.info(f'Loss/svrnn: {svrnn_loss.item()}')
    writer.add_scalar(f'Loss/SVRNN', svrnn_loss.item(), global_step=i)

    svlstm_output_, _ = svlstm(sequence)
    svlstm_output = svlstm.fit_dim(svlstm_output_)
    svlstm_loss = criterion(svlstm_output.squeeze(0)[-memory_lenght:], target)

    svlstm_optimizer.zero_grad()
    svlstm_loss.backward()
    svlstm_optimizer.step()
    # svlstm_scheduler.step()

    logging.info(f'Loss/svlstm: {svlstm_loss.item()}')
    writer.add_scalar(f'Loss/SVLSTM', svlstm_loss.item(), global_step=i)

    svgru_output_, _ = svgru(sequence)
    svgru_output = svgru.fit_dim(svgru_output_)
    svgru_loss = criterion(svgru_output.squeeze(0)[-memory_lenght:], target)

    svgru_optimizer.zero_grad()
    svgru_loss.backward()
    svgru_optimizer.step()
    # svgru_scheduler.step()

    logging.info(f'Loss/svgru: {svgru_loss.item()}')
    writer.add_scalar(f'Loss/SVGRU', svgru_loss.item(), global_step=i)

    mvrnn_output, _ = mvrnn(sequence)
    # mvrnn_output = mvrnn.fit_dim(mvrnn_output_)
    mvrnn_loss = criterion(mvrnn_output.squeeze(0)[-memory_lenght:], target)

    mvrnn_optimizer.zero_grad()
    mvrnn_loss.backward()
    mvrnn_optimizer.step()
    # mvrnn_scheduler.step()

    logging.info(f'Loss/mvrnn: {mvrnn_loss.item()}')
    writer.add_scalar(f'Loss/MVRNN', mvrnn_loss.item(), global_step=i)

    mvlstm_output, _ = mvlstm(sequence)
    # mvlstm_output = mvlstm.fit_dim(mvlstm_output_)
    mvlstm_loss = criterion(mvlstm_output.squeeze(0)[-memory_lenght:], target)

    mvlstm_optimizer.zero_grad()
    mvlstm_loss.backward()
    mvlstm_optimizer.step()
    # mvlstm_scheduler.step()

    logging.info(f'Loss/mvlstm: {mvlstm_loss.item()}')
    writer.add_scalar(f'Loss/MVLSTM', mvlstm_loss.item(), global_step=i)

    mvgru_output, _ = mvgru(sequence)
    # mvgru_output = mvgru.fit_dim(mvgru_output_)
    mvgru_loss = criterion(mvgru_output.squeeze(0)[-memory_lenght:], target)

    mvgru_optimizer.zero_grad()
    mvgru_loss.backward()
    mvgru_optimizer.step()
    # mvgru_scheduler.step()

    logging.info(f'Loss/mvgru: {mvgru_loss.item()}')
    writer.add_scalar(f'Loss/MVGRU', mvgru_loss.item(), global_step=i)

    drnn_output, _ = drnn(sequence)
    drnn_loss = criterion(drnn_output[-1][-memory_lenght:].squeeze(1), target)

    drnn_optimizer.zero_grad()
    drnn_loss.backward()
    drnn_optimizer.step()
    # drnn_scheduler.step()

    logging.info(f'Loss/DRNN: {drnn_loss.item()}')
    writer.add_scalar(f'Loss/DRNN', drnn_loss.item(), global_step=i)

    dlstm_output, _ = dlstm(sequence)
    dlstm_loss = criterion(dlstm_output[-1][-memory_lenght:].squeeze(1), target)

    dlstm_optimizer.zero_grad()
    dlstm_loss.backward()
    dlstm_optimizer.step()
    # dlstm_scheduler.step()

    logging.info(f'Loss/DLSTM: {dlstm_loss.item()}')
    writer.add_scalar(f'Loss/DLSTM', dlstm_loss.item(), global_step=i)

    dgru_output, _ = dgru(sequence)
    dgru_loss = criterion(dgru_output[-1][-memory_lenght:].squeeze(1), target)

    dgru_optimizer.zero_grad()
    dgru_loss.backward()
    dgru_optimizer.step()
    # dgru_scheduler.step()

    logging.info(f'Loss/DGRU: {dgru_loss.item()}')
    writer.add_scalar(f'Loss/DGRU', dgru_loss.item(), global_step=i)

    msrnn_output, _ = msrnn(sequence)
    msrnn_loss = criterion(msrnn_output[-1][-memory_lenght:].squeeze(1), target)

    msrnn_optimizer.zero_grad()
    msrnn_loss.backward()
    msrnn_optimizer.step()
    # msrnn_scheduler.step()

    logging.info(f'Loss/MSRNN: {msrnn_loss.item()}')
    writer.add_scalar(f'Loss/MSRNN', msrnn_loss.item(), global_step=i)

    mslstm_output, _ = mslstm(sequence)
    mslstm_loss = criterion(mslstm_output[-1][-memory_lenght:].squeeze(1), target)

    mslstm_optimizer.zero_grad()
    mslstm_loss.backward()
    mslstm_optimizer.step()
    # mslstm_scheduler.step()

    logging.info(f'Loss/MSLSTM: {mslstm_loss.item()}')
    writer.add_scalar(f'Loss/MSLSTM', mslstm_loss.item(), global_step=i)

    msgru_output, _ = msgru(sequence)
    msgru_loss = criterion(msgru_output[-1][-memory_lenght:].squeeze(1), target)

    msgru_optimizer.zero_grad()
    msgru_loss.backward()
    msgru_optimizer.step()
    # msgru_scheduler.step()

    logging.info(f'Loss/MSGRU: {msgru_loss.item()}')
    writer.add_scalar(f'Loss/MSGRU', msgru_loss.item(), global_step=i)

    writer.add_scalars(f'Loss/copy_memory_problem', {
        'SVRNN': svrnn_loss.item(),
        'SVLSTM': svlstm_loss.item(),
        'SVGRU': svgru_loss.item(),
        'MVRNN': mvrnn_loss.item(),
        'MVLSTM': mvlstm_loss.item(),
        'MVGRU': mvgru_loss.item(),
        'DRNN': drnn_loss.item(),
        'DLSTM': dlstm_loss.item(),
        'DGRU': dgru_loss.item(),
        'MSRNN': msrnn_loss.item(),
        'MSLSTM': mslstm_loss.item(),
        'MSGRU': msgru_loss.item()
    }, global_step=i)

    if i % 100 == 0:
        print(f'Iteration: {i}, SVRNN loss: {round(svrnn_loss.item(), 2)}, SVLSTM loss: {round(svlstm_loss.item(), 2)}, SVGRU loss: {round(svgru_loss.item(), 2)}, MVRNN loss: {round(mvrnn_loss.item(), 2)}, MVLSTM loss: {round(mvlstm_loss.item(), 2)}, MVGRU loss: {round(mvgru_loss.item(), 2)}, DRNN loss: {round(drnn_loss.item(), 2)}, DLSTM loss: {round(dlstm_loss.item(), 2)}, DGRU loss: {round(dgru_loss.item(), 2)}, MSRNN loss: {round(msrnn_loss.item(), 2)}, MSLSTM loss: {round(mslstm_loss.item(), 2)}, MSGRU loss: {round(msgru_loss.item(), 2)}')
