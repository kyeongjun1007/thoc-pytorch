import pandas as pd

ts = pd.read_csv('./data/TimeSeries.csv')
tsl = pd.read_csv('./data/labelsTimeSeries.csv')

ts['label'] = tsl

ts.to_csv('ts_dataset.csv', header=True, index = False)


import torch

(torch.randn(3,10) < 0.5).max()

torch.tensor([1, 1, 0]) > 0


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./data/PowerDemand.csv')

plt.plot(df.index, df.iloc[:,0])
plt.show()

from gridsearch import get_param_list
from itertools import product
for i,v in enumerate(product(*get_param_list('kddcup99'))) :
    print(i)

from torch.optim.lr_scheduler import LambdaLR
import torch.optim.lr_scheduler.LambdaLR

5%5

'./powerdemand.csv/1234'.split('/')[1]