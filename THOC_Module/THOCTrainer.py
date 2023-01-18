from THOCBase import THOCBase
from torch.optim import Adam as Optimizer
from logging import getLogger
import pandas as pd
import numpy as np
import torch
import datetime


class THOCTrainer(THOCBase):
    def __init__(self, model_params, logger_params, run_params, optimizer_params):
        super(THOCTrainer, self).__init__(model_params, logger_params, run_params)

        self.optimizer = Optimizer(self.model.parameters(), **optimizer_params)

    def run(self):
        print('**************** Training Start ****************')
        for epoch in range(self.run_params['epochs']):
            train_loss = self._train_one_epoch()
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                  ". Epochs : %d, loss : %1.5f" % (epoch, train_loss))

        print('**************** Training Start ****************')
        print('**************** Training Done ****************')

    def _train_one_epoch(self):
        train_dataloader = self.data_configurator.train_dataloader()
        for i, window in enumerate(train_dataloader):
            if i == 0:
                pass
            window = window.to(self.device)

            anomaly_scores, centroids_diff, out_of_drnn = self.model.forward(window)

            loss = self._get_loss(anomaly_scores, centroids_diff, out_of_drnn, window)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()
