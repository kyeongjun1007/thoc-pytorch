#
# class aaa() :
#     def __init__(self, a, b):
#         self.a = a
#         self.b = b
#         self.run()
#     #
#     # def init_c(self, c):
#     #     self.c = c
#
#     def run(self):
#
#         return self.a+self.b#+self.c
#
# a = aaa(1,3)
# a.init_c(4)
# a.run()

from DataConfigurator import DataConfigurator
from THOCBase import THOC
import torch

batch_size = 32
window_size = 80
file_name = './PowerDemand.csv'
valid_ratio = 0.3
test_ratio = 0.2

train_params = {
        'use_cuda' : True,
        'cuda_device_num' : 0,
        'epochs' : 1,
        'batch_size' : batch_size,
        'window_size' :window_size,
        'lambda_l2reg' : 1e-06,
        'lambda_orth' : 1e-02,
        'lambda_orth' : 1e-02,
        'lr' : 1e-03,
        'data_config' : {
            'file_name' : file_name,
            'valid_ratio' : valid_ratio,
            'test_ratio': test_ratio,
            'header' : 0,
            'index_col' : False,
            'shuffle' : False
        }
}

model_params = {
        'n_input' : 1,
        'n_hidden' : 9,
        'n_layers' : 3,
        'n_centroids' : [6,4,3],
        'tau' : 0.01,
        'dropout' : 0,
        'cell_type' : 'RNN',
        'batch_first' : True
    }
device = "cuda:0"

dc = DataConfigurator(train_params)
init_data = dc.get_init_data().to(device)


model = THOC(**model_params, init_data=init_data).to(device)

from torch.optim import Adam as Optimizer

optimizer_params = {
        'lr' : 1e-03,
        'weight_decay' : 1e-05
    }

om = Optimizer(model.parameters(), **optimizer_params)

print(om)

i = 0
while True :
    try :
        print(i)
        break
    except :
        pass
    i += 1