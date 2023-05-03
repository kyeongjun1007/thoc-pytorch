import torch
import numpy as np
import random
from torch.utils.data import Dataset

"""
scenario 1 : increasing amplitude
scenario 2 : decreasing amplitude
scenario 3 : randomly choice one signal for 4 times
scenario 4 : randomly choice two signals and add them for 4 times
scenario 5 : randomly choice three signals and add them for 4 times
"""


class DataGenerator :
    def __init__(self, amplitude, frequency, phase, t) :
        self.signals = [amp * torch.sin(2 * torch.pi * freq * t + p) for amp, freq, p in zip(amplitude, frequency, phase)]
        self.noise = [torch.from_numpy(np.random.normal(0, 0.1*amp, signal.shape)).float() for amp, signal in zip(amplitude, self.signals)]
        self.noisy_signals = [signal + noise for signal, noise in zip(self.signals, self.noise)]

        self.scenario1 = torch.cat(((self.noisy_signals[0] + self.noisy_signals[1]),
            (self.noisy_signals[0] + self.noisy_signals[1] + self.noisy_signals[2]),
            (self.noisy_signals[1] + self.noisy_signals[2] + self.noisy_signals[3]),
            (self.noisy_signals[2] + self.noisy_signals[3])
        ))
        self.scenario2 = torch.cat((
            (self.noisy_signals[2] + self.noisy_signals[3]),
            (self.noisy_signals[1] + self.noisy_signals[2] + self.noisy_signals[3]),
            (self.noisy_signals[0] + self.noisy_signals[1] + self.noisy_signals[2]),
            (self.noisy_signals[0] + self.noisy_signals[1])
        ))
        self.scenario3 = torch.cat(
            random.sample(self.noisy_signals, 4)
        )
        self.scenario4 = torch.cat((
            torch.stack(random.sample(self.noisy_signals, 2)).sum(0),
            torch.stack(random.sample(self.noisy_signals, 2)).sum(0),
            torch.stack(random.sample(self.noisy_signals, 2)).sum(0),
            torch.stack(random.sample(self.noisy_signals, 2)).sum(0)
        ))
        self.scenario5 = torch.cat((
            torch.stack(random.sample(self.noisy_signals, 3)).sum(0),
            torch.stack(random.sample(self.noisy_signals, 3)).sum(0),
            torch.stack(random.sample(self.noisy_signals, 3)).sum(0),
            torch.stack(random.sample(self.noisy_signals, 3)).sum(0)
        ))

    def get_data(self, scenario_num) :
        if scenario_num == '1' :
            return self.scenario1
        elif scenario_num == '2' :
            return self.scenario2
        elif scenario_num == '3' :
            return self.scenario3
        elif scenario_num == '4' :
            return self.scenario4
        elif scenario_num == '5' :
            return self.scenario5
        else :
            raise ValueError('Invalid scenario number')


class SlidingWindow(Dataset):
    def __init__(self, X, window):
        self.X = X
        self.window = window

    def __getitem__(self, index):
        X = self.X[index:index + self.window]
        y = self.X[index + 1:index + self.window +1]
        return X, y

    def __len__(self):
        return len(self.X) - self.window - 1