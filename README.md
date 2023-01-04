# review_models
## THOC
Dilated RNN model code refers to [Zalando Research](https://github.com/zalandoresearch/pytorch-dilated-rnn).   
And I changed the output of this model to extract whole layer outs.

### THOC.py
pytorch implementation of [THOC](https://proceedings.neurips.cc/paper/2020/file/97e401a02082021fd24957f852e0e475-Paper.pdf) model.   
It should be modified so that batches are calculated in one go    
배치 다시 수정 ㅠㅠ

### THOC_example.py
Usage example of THOC with sliding window.   
You can train this model for mini-batch learning and use GPU.   

### THOC_debugging.py
For debugging...

## SRNN   
### WRNN.py   
Scaled-RNN version 1 trial.   
version1 : make scaled_input tensor using mean and predict multi-scaled y_hat.   
Scaled-RNN class를 따로 만들고 multi-layer를 stack 하는 방식으로 다시 짜야 할 듯.
