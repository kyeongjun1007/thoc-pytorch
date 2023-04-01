# review_models
## THOC
Dilated RNN model code refers to [Zalando Research](https://github.com/zalandoresearch/pytorch-dilated-rnn).   
And I changed the output of this model to extract whole layer outs.   
   
### THOC.py
pytorch implementation of [THOC](https://proceedings.neurips.cc/paper/2020/file/97e401a02082021fd24957f852e0e475-Paper.pdf) model.   
You can train this model using your GPU. (So that batches are calculated in one go)   
   
### experiment_mobiacts.py
Usage example of THOC with open-dataset(MobiActs).   
   
## THOC_Module
Make THOC module. it contains Trainer(train and valid), DataConfigurator(data and model config), Base(build model and sliding window etc...)   
some code refers to Yeong-Dae Kwon

23.02.10
learning rate is decayed after 20 epochs and 1/tau is increaseed every 5 epochs
pytorch_kmeans is revised (can operate when num_clusters = 1)

The MIT License

Copyright (c) 2021 Yeong-Dae Kwon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.   
   
## THOC_Module_seperateddata
This allows you to experiment THOC model if your dataset is seperated with 3 parts (train, valid, test).   
the three datasets have to be saved in data folder and named '(dataname)_(train or valid or test).csv'

## MSRNN   
Multi Scale RNN model code.   

### MSRNN.py
pytorch Implementation of MSRNN.   
RNN can be alternated to LSTM or GRU.   


## SRNN   
RNN with another scaled input data.   
It can predict single/multiple time point.   
SRNN_s means single time point prediction and SRNN_m means multiple time point prediction.   
VRNN is vanilla RNN model for comparison. (_s and _m means same)

### (Model)_(p).py   
Pytorch implementation of Models
(Model) : SRNN(scaled input version RNN) / VRNN(vanilla RNN)
(p) : s(single point prediction) / m(multiple point prediction)

### gridsearch_(model)_(p).py
Grid-search of models.   
It makes log(parameters and loss), scalers(pickle), model.pt(parameters of best model) files
