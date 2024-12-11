# thoc-pytorch
## Repository Overview
this is pytorch implementation of [THOC](https://proceedings.neurips.cc/paper/2020/file/97e401a02082021fd24957f852e0e475-Paper.pdf) model (unofficial).   
Dilated RNN model code refers to [Zalando Research](https://github.com/zalandoresearch/pytorch-dilated-rnn).   
And I changed the output of this model to extract whole layer outs.   
   
## repo 파일 구조
<pre>
thoc-pytorch/   
├── DataConfigurator.py         # train-valid-test split, DataLoader 설정, test 데이터 저장 등.   
├── THOC.py                     # THOC 모델 Class.   
├── THOCBase.py                 # 학습 과정 log 및 tensor board 저장, best model 저장 등.   
├── THOCTrainer.py              # THOC 모델 학습.   
├── gridsearch.py               # grid-search scope 설정.   
├── train.py                    # 모델 학습 실행 코드.   
├── test.py                     # best 모델의 test 성능 확인 코드. (수정 중)   
├── README.md   
└── requirements.txt
</pre>
