from THOC import THOC
from DataConfigurator import DataConfigurator

import torch

from pathlib import Path
from logging import getLogger
from common.utils import get_result_folder, LogData


class THOCBase:
    def __init__(self, model_params, logger_params, run_params):
        self.model_params = model_params
        self.logger_params = logger_params
        self.run_params = run_params

        self.device = torch.device("cuda", run_params['cuda_device_num'] if run_params['use_cuda'] else "cpu")

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder(logger_params['log_file']['desc'],
                                               date_prefix=logger_params['log_file']['date_prefix'],
                                               result_dir=logger_params['log_file']['result_dir']
                                               )
        Path(self.result_folder).mkdir(parents=True, exist_ok=True)

        self.result_log = LogData()

        # data config
        self.data_configurator = DataConfigurator(self.run_params, self.result_folder)
        init_data = self.data_configurator.get_init_data(run_params['use_cuda'])

        # model
        self.model = THOC(**self.model_params, init_data=init_data).to(self.device)

    def _get_loss(self, anomaly_scores, centroids_diff, out_of_drnn, timeseries_input):

        l2_loss = []
        for n, p in self.model.named_parameters():
            if n == "cluster_centers":
                for i in range(len(p)):
                    l2_loss.append(sum(torch.linalg.norm(p_[i], 2) for p_ in p))
            else:
                l2_loss.append(torch.linalg.norm(p, 2))
        loss_l2reg = torch.tensor(l2_loss).sum()

        loss_thoc = anomaly_scores.mean() + self.run_params['lambda_l2reg'] * loss_l2reg

        loss_orth = centroids_diff.mean()

        tss_list = []
        for i, out in enumerate(out_of_drnn):
            dilation = 2 ** i
            tss = ((out[:, :-dilation] - timeseries_input[:, dilation:]) ** 2).mean()
            tss_list.append(tss)
        loss_tss = torch.stack(tss_list).mean()

        loss = loss_thoc + loss_orth * self.run_params['lambda_orth'] + loss_tss * self.run_params['lambda_tss']

        return loss

    def _log(self, epoch, epochs, train_loss, valid_loss, best_loss, early_stop):
        if not early_stop:
            self.logger.info('Epoch: %d/%d, train_loss: %1.5f, valid_loss: %1.5f, best_loss: %1.5f' % (
                epoch, epochs, train_loss, valid_loss, best_loss))
        else:
            self.logger.info('Early stoped. because validation loss did not improve for a long time.')

    def _save_params(self):
        param_dict = self.model.state_dict()
        torch.save(param_dict, f'{self.result_folder}/param_dict.pt')