# debug code
# pip install KMeans-pytorch
import os
import THOC
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

# data setting

os.chdir("C:/Users/IDSL/Desktop/aaa")

data = torch.randn(30,1,9)


# model setting

n_input = 9
n_hidden = 9
n_layers = 3
cell_type = 'RNN'
dropout = 0
n_centroids = [6,4,3]

use_cuda = False

model = THOC(n_input, n_hidden, n_layers, n_centroids, dropout = dropout, cell_type = cell_type)

model.forward(data, first = True)




# pip install KMeans-pytorch
import os
import THOC
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

# data setting

os.chdir("C:/Users/Kyeongjun/Desktop/water_level_pred")

data = pd.read_csv("file.csv")

class SlidingWindow(Dataset) :
    def __init__(self, data, window) :
        self.data = data
        self.window = window
        
    def __getitem__(self, index) :
        x = self.data[index:index+self.window]
        return x
    
    def __len__(self) :
        return len(self.data) - self.window

window_size = 144

# train_test_split (8:2)
k = int(len(data)*0.8)

train = data.iloc[:k,:]
test = data.iloc[k:,:]

# sliding window에 맞게 데이터 조정
window_size = 144

# DataLoader 생성
train = torch.tensor(np.array(train, dtype=float))
test = torch.tensor(np.array(test, dtype=float))

train_dataset = SlidingWindow(data=train, window=144)
train_dl = DataLoader(dataset=train_dataset,
           batch_size=1,
           shuffle=False)

test_dataset = SlidingWindow(data=test, window=144)
test_dl = DataLoader(dataset=test_dataset,
           batch_size=1,
           shuffle=False)

# loss function
def thoc_loss() :
    loss_toch = torch.matmul(torch.t(R),(1-self.cos(f_hat,self.cluster_centers[-1])))/KL            # sum(R*d)/K^L
    
    co = torch.matmul(torch.t(self.cluster_centers),self.cluster_centers) -
         torch.eye(self.cluster_centers.size(0))                                                # co = t(C)*C-I
    co = np.linalg.norm(co, ord='fro')                                                          # co = frobenius norm of co
    loss_orth = (co*co)/KL                                                                      # co^2/the num of last layer centroids
    
    loss_tss = 
    
    loss = loss_thoc + self.lambda_orth*loss_orth + self.lambda_tss*loss_tss                                                     # 최종 loss
   
    pass

# model setting

n_input = 9
n_hidden = 9
n_layers = 3
cell_type = 'RNN'
dropout = 0

use_cuda = torch.cuda.is_available()

model = THOC(n_input, n_hidden, n_layers, dropout = dropout, cell_type = cell_type)

num_epochs = 100
learning_rate = 0.001

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training
for epoch in range(num_epochs) :
    for i, window in enumerate(train_dl):
        window = window.type(torch.float32)
        window = Variable(window)
        optimizer.zero_grad()
        outputs = model.forward(window)
        loss = thoc_loss()
        loss.backward()
    
        optimizer.step()
        if i % 30 == 0 :
            print("window steps : %d, loss : %1.5f" %(i, loss.item()))
    print("Epochs : %d, loss : %1.5f" %(epoch, loss.item()))
    
    