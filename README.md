# review_models
## THOC
Dilated RNN model code refers to [Zalando Research](https://github.com/zalandoresearch/pytorch-dilated-rnn).   
And I changed the output of this model to extract whole layer outs.

### THOC.py
pytorch implementation of [THOC](https://proceedings.neurips.cc/paper/2020/file/97e401a02082021fd24957f852e0e475-Paper.pdf) model.   
<<<<<<< HEAD
You can train this model using your GPU. (Batches are calculated in one go)
   
### THOC_example.py
Usage example of THOC with sliding window.   
You can train this model for mini-batch learning and use GPU.   

### THOC_debugging.py
For debugging...
=======
You can train this model for mini-batch learning and use GPU. (So that batches are calculated in one go)    
And If you don't use scaler for your data, there might be OOM error   
becuase the tanh gate of vanilla RNN make your model output same

### experiment_mobiacts.py
Usage example of THOC with open-dataset(MobiActs).   
>>>>>>> 867e2d67b6a5cd95708eef7a0e46bf2ec5065425

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
