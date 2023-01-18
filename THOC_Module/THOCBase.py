from THOC import THOC
from DataConfigurator import DataConfigurator

import torch


class THOCBase:
    def __init__(self, model_params, logger_params, run_params):
        self.model_params = model_params
        self.logger_params = logger_params
        self.run_params = run_params

        self.device = torch.device("cuda", run_params['cuda_device_num'] if run_params['use_cuda'] else "cpu")

        # data config
        self.data_configurator = DataConfigurator(**self.run_params)
        init_data = self.data_configurator.get_init_data()

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
            for b in range(out_of_drnn.shape[2]):
                tss = ((out[:-dilation, b] - timeseries_input[b, dilation:]) ** 2).mean()
                tss_list.append(tss)
        loss_tss = torch.stack(tss_list).mean()

        loss = loss_thoc + loss_orth * self.run_params['lambda_orth'] + loss_tss * self.run_params['lambda_tss']

        return loss
