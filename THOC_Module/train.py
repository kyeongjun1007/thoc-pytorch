import random
import numpy as np
import torch

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CIDA_DEVICE_NUM = 0

from common.utils import create_logger
from THOCTrainer import THOCTrainer

def check_debug() :
    import sys

    eq = sys.gettrace() is None

    if eq is False :
        return True
    else :
        return False


if __name__ == '__main__' :
    batch_size = 32
    window_size = 80
    file_name = './~~~'
    valid_ratio = 0.3
    test_ratio = 0.2
    # random.seed(1)
    # torch.manual_seed(1)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(1)

    model_params = {
        'n_input' : 9,
        'n_hidden' : 9,
        'n_layers' : 3,
        'n_centroids' : [6,4,3],
        'tau' : 0.01,
        'dropout' : 0,
        'cell_type' : 'RNN',
        'batch_first' : True
    }

    logger_params = {
        'log_file' : {
            'result_dir' : './result/',
            'filename' : 'log.txt',
            'date_prefix' : False
        }
    }

    train_params = {
        'use_cuda' : True,
        'cuda_device_num' : 0,
        'epochs' : 1,
        'batch_size' : batch_size,
        'window_size' :window_size,
        'lambda_l2reg' : 1e-06,
        'lambda_orth' : 1e-02,
        'lambda_orth' : 1e-02,
        'data_config' : {
            'file_name' : file_name,
            'valid_ratio' : valid_ratio,
            'test_ratio': test_ratio,
            'header' : 0,
            'index_col' : False,
            'shuffle' : False
        }
        # 'logging' : {
        #     'model_save_interval' : 500,
        #     'log_interval' : 1,
        # }
    }

    optimizer_params = {
        'lr' : 1e-03,
        'weight_decay' : 1e-05
    }
    create_logger(**logger_params)

    trainer = THOCTrainer(model_params=model_params,
                          logger_params=logger_params,
                          run_params=train_params,
                          optimizer_params=optimizer_params)

    trainer.run()