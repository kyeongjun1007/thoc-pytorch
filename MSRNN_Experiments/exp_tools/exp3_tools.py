import torch
from torch.autograd import Variable
import numpy as np


def sequence_generator(T, num_components) :
    seq = np.arange(num_components)
    np.random.shuffle(seq)

    sequence = torch.from_numpy(seq).float()
    sequence = torch.repeat_interleave(sequence, T // num_components, dim=0)
    answer = torch.unique(sequence)

    return seq.unsqueeze(0), answer.unsqueeze(0)


def data_generator(T, mem_length):

    seq = torch.from_numpy(np.random.randint(0, 8, size=(mem_length,))).float()
    blanck = 8 * torch.ones(T)
    marker = 9 * torch.ones(mem_length + 1)
    placeholders = 8 * torch.ones(mem_length)

    x = torch.cat((seq, blanck[:-1], marker), 0)
    y = torch.cat((placeholders, blanck, seq), 0).long()

    x, y = Variable(x), Variable(y)
    return x.unsqueeze(0), y.unsqueeze(0)