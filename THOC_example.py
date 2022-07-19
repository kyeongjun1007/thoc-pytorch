# pip install KMeans-pytorch
import THOC
import torch
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import os

os.chdir("C:/Users/user/Desktop/datasets")

n_input = 9
n_hidden = 2
n_layers = 3
cell_type = 'GRU'

data = pd.read_csv("power_demand.csv")
data = torch.tensor(np.array(data))
data = data.view(438,80,1)          # batch_size = 80으로 data slicing

init_data = data[0]
data = data[1:]

