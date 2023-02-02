import random
import numpy as np
import torch

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

from common.utils import create_logger
from THOCTrainer import THOCTrainer


def check_debug():
    import sys

    eq = sys.gettrace() is None

    if eq is False:
        return True
    else:
        return False


if __name__ == '__main__':
    batch_size = 32
    window_size = 80
    data_dir = './data/'
    data_name = 'TS_dataset.csv'
    valid_ratio = 0.3
    test_ratio = 0.2
    random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(1)

    model_params = {
        'n_input': 11,
        'n_hidden': 32,
        'n_layers': 3,
        'n_centroids': [6, 4, 3],
        'tau': 0.1,
        'dropout': 0,
        'cell_type': 'RNN',
        'batch_first': True
    }

    logger_params = {
        'log_file': {
            'result_dir': './result/',
            'desc': f"./{data_name}",
            'filename': 'log.txt',
            'date_prefix': False
        }
    }

    train_params = {
        'use_cuda': True,
        'cuda_device_num': 0,
        'epochs': 10,
        'max_loss_maintain': 5,
        'batch_size': batch_size,
        'window_size': window_size,
        'lambda_l2reg': 1e-06,
        'lambda_orth': 1e-02,
        'lambda_tss': 1e-02,
        'data_config': {
            'data_dir': data_dir,
            'data_name': data_name,
            'label': True,
            'valid_ratio': valid_ratio,
            'test_ratio': test_ratio,
            'header': 0,
            'index_col': False,
            'shuffle': False
        },
        'logging': {
            'log_image_params': {
                'json_foldername': 'log_image_style',
                'filename': 'style_loss.json'
            }

        }
    }

    optimizer_params = {
        'lr': 1e-02,
        'weight_decay': 0.65
    }

    create_logger(**logger_params)

    trainer = THOCTrainer(model_params=model_params,
                          logger_params=logger_params,
                          run_params=train_params,
                          optimizer_params=optimizer_params)

    trainer.run()

else :
    batch_size = 32
    window_size = 80
    data_dir = './data/'
    data_name = 'TS_dataset.csv'
    valid_ratio = 0.3
    test_ratio = 0.2
    random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(1)

    model_params = {
        'n_input': 11,
        'n_hidden': 32,
        'n_layers': 3,
        'n_centroids': [6, 4, 3],
        'tau': 1,
        'dropout': 0,
        'cell_type': 'RNN',
        'batch_first': True
    }

    logger_params = {
        'log_file': {
            'result_dir': './result/',
            'desc': f"./{data_name}",
            'filename': 'log.txt',
            'date_prefix': False
        }
    }

    train_params = {
        'use_cuda': True,
        'cuda_device_num': 0,
        'epochs': 10,
        'max_loss_maintain': 5,
        'batch_size': batch_size,
        'window_size': window_size,
        'lambda_l2reg': 1e-06,
        'lambda_orth': 1e-02,
        'lambda_tss': 1e-02,
        'data_config': {
            'data_dir': data_dir,
            'data_name': data_name,
            'label': True,
            'valid_ratio': valid_ratio,
            'test_ratio': test_ratio,
            'header': 0,
            'index_col': False,
            'shuffle': False
        },
        'logging': {
            'log_image_params': {
                'json_foldername': 'log_image_style',
                'filename': 'style_loss.json'
            }

        }
    }

    optimizer_params = {
        'lr': 1e-02,
        'weight_decay': 0.65
    }

    def get_test_params():
        test_params = {
            'batch_size': batch_size,
            'window_size': window_size,
            'shuffle': train_params['data_config']['shuffle'],
            'header' : train_params['data_config']['header'],
            'index_col': train_params['data_config']['index_col'],

            'model_params': model_params,
            'logger_params': logger_params,

            'init_data': torch.randn(batch_size, window_size, model_params['n_input']) if train_params[
                'use_cuda'] else torch.randn(window_size, batch_size, model_params['n_input']),
            'data_name': f"./{data_name}"[:-4] + '_test.csv',
            'test_params': {  #
                'lambda_l2reg': 1e-06,
                'lambda_orth': 1e-02,
                'lambda_tss': 1e-02
            },
            'threshold': 1
        }

        return test_params
