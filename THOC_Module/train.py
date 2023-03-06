import random
import numpy as np
import torch
from itertools import product
from gridsearch import get_param_list
from pathlib import Path

# DEBUG_MODE = False
# USE_CUDA = not DEBUG_MODE
# CUDA_DEVICE_NUM = 1

from common.utils import create_logger
from THOCTrainer import THOCTrainer


# def check_debug():
#     import sys
#
#     eq = sys.gettrace() is None
#
#     if eq is False:
#         return True
#     else:
#         return False


if __name__ == '__main__':

    data_name = 'powerdemand.csv'
    result_dir = '/result/'
    Path(f'{result_dir}{data_name}/best').mkdir(parents=True, exist_ok=True)

    best_loss = float('inf')

    for i, hyperparams in enumerate(product(*get_param_list(data_name))) :

        random.seed(1)
        torch.manual_seed(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(1)

        max_loss_maintain, n_hidden, n_centroids, tau, batch_size, skip_length, lambda_orth, lambda_tss, lr = hyperparams
        print(hyperparams)
        window_size = 80
        n_input = 1
        header = 0
        index_col = False
        shuffle = True
        label = False

        data_dir = './data/'
        valid_ratio = 0.3
        test_ratio = 0.2

        model_params = {
            'n_input': n_input,
            'n_hidden': n_hidden,
            'n_layers': 3,
            'n_centroids': n_centroids,
            'tau': tau,
            'tau_decay' : 0.9,
            'dropout': 0,
            'cell_type': 'RNN',
            'batch_first': True
        }

        logger_params = {
            'log_file': {
                'result_dir': './result/',
                'desc': f"./{data_name}/{i}",
                'filename': 'log.txt',
                'date_prefix': False
            }
        }

        train_params = {
            'use_cuda': True,
            'cuda_device_num': 0,
            'epochs': 100,
            'max_loss_maintain': max_loss_maintain,
            'batch_size': batch_size,
            'window_size': window_size,
            'lambda_l2reg': 1e-06,
            'lambda_orth': lambda_orth,
            'lambda_tss': lambda_tss,
            'data_config': {
                'data_dir': data_dir,
                'data_name': data_name,
                'label': label,
                'valid_ratio': valid_ratio,
                'test_ratio': test_ratio,
                'header': header,
                'index_col': index_col,
                'shuffle': shuffle
            },
            'logging': {
                'log_image_params': {
                    'json_foldername': 'log_image_style',
                    'filename': 'style_loss.json'
                }

            }
        }

        optimizer_params = {
            'lr': lr,
            'lr_decay': 0.65,
            'decay_step' : 20
        }

        create_logger(**logger_params)

        trainer = THOCTrainer(model_params=model_params,
                              logger_params=logger_params,
                              run_params=train_params,
                              optimizer_params=optimizer_params,
                              )

        loss, param_dict = trainer.run(hyperparams)

        if best_loss > loss :
            best_loss = loss
            torch.save(param_dict, f'{result_dir}{data_name}/best/param_dict.pt')


# else :
#     batch_size = 32
#     window_size = 80
#     data_dir = './data/'
#     data_name = 'TS_dataset.csv'
#     valid_ratio = 0.3
#     test_ratio = 0.2
#     random.seed(1)
#     torch.manual_seed(1)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     np.random.seed(1)
#
#     model_params = {
#         'n_input': 11,
#         'n_hidden': 32,
#         'n_layers': 3,
#         'n_centroids': [6, 4, 3],
#         'tau': 1,
#         'dropout': 0,
#         'cell_type': 'RNN',
#         'batch_first': True
#     }
#
#     logger_params = {
#         'log_file': {
#             'result_dir': './result/',
#             'desc': f"./{data_name}",
#             'filename': 'log.txt',
#             'date_prefix': False
#         }
#     }
#
#     train_params = {
#         'use_cuda': True,
#         'cuda_device_num': 0,
#         'epochs': 100,
#         'max_loss_maintain': 10,
#         'batch_size': batch_size,
#         'window_size': window_size,
#         'lambda_l2reg': 1e-06,
#         'lambda_orth': 1e-02,
#         'lambda_tss': 1e-02,
#         'data_config': {
#             'data_dir': data_dir,
#             'data_name': data_name,
#             'label': True,
#             'valid_ratio': valid_ratio,
#             'test_ratio': test_ratio,
#             'header': 0,
#             'index_col': False,
#             'shuffle': False
#         },
#         'logging': {
#             'log_image_params': {
#                 'json_foldername': 'log_image_style',
#                 'filename': 'style_loss.json'
#             }
#
#         }
#     }
#
#     optimizer_params = {
#         'lr': 1e-02,
#         'weight_decay': 0.65
#     }
#
#     def get_test_params():
#         test_params = {
#             'batch_size': 1,
#             'window_size': window_size,
#             'shuffle': train_params['data_config']['shuffle'],
#             'header' : train_params['data_config']['header'],
#             'index_col': train_params['data_config']['index_col'],
#
#             'model_params': model_params,
#             'logger_params': logger_params,
#
#             'init_data': torch.randn(batch_size, window_size, model_params['n_input']) if train_params[
#                 'use_cuda'] else torch.randn(window_size, batch_size, model_params['n_input']),
#             'data_name': f"./{data_name}"[:-4] + '_test.csv',
#             'test_params': {  #
#                 'lambda_l2reg': 1e-06,
#                 'lambda_orth': 1e-02,
#                 'lambda_tss': 1e-02
#             },
#             'threshold': 0.52
#         }
#
#         return test_params
