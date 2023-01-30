from THOCBase import THOCBase
from torch.optim import Adam as Optimizer
from logging import getLogger
import pandas as pd
import numpy as np
import torch
import datetime

from common.utils import util_save_log_image_with_label


class THOCTrainer(THOCBase):
    def __init__(self, model_params, logger_params, run_params, optimizer_params):
        super(THOCTrainer, self).__init__(model_params, logger_params, run_params)

        self.optimizer = Optimizer(self.model.parameters(), **optimizer_params)

        self.loss_best = float("inf")
        self.loss_maintain = 0

    def run(self):

        print('**************** Training Start ****************')
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        for epoch in range(self.run_params['epochs']):
            # Training and Validation
            train_loss = self._train_one_epoch()
            valid_loss = self._valid_one_epoch()

            self.result_log.append('train_loss', epoch, train_loss)
            self.result_log.append('valid_loss', epoch, valid_loss)

            if valid_loss < self.loss_best:
                self.loss_best = valid_loss
                self.loss_maintain = 0
            else:
                self.loss_maintain += 1

            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                  ". Epochs : %d, train_loss : %1.5f, valid_loss : %1.5f" % (epoch, train_loss, valid_loss))

            # Logging
            self._log(epoch, self.run_params['epochs'], train_loss, valid_loss, self.loss_best, early_stop=False)
            if self.loss_maintain >= (self.run_params['max_loss_maintain']+1):
                self._log(epoch, self.run_params['epochs'], train_loss, valid_loss, self.loss_best, early_stop=True)
                break

        # Save model params
        self._save_params()

        # Save Loss Images
        image_prefix = f'{self.result_folder}/img'
        util_save_log_image_with_label(image_prefix, self.run_params['logging']['log_image_params'],
                                       self.result_log, labels=['train_loss', 'valid_loss'])

        print('**************** Training Done ****************')

    def _train_one_epoch(self):
        train_dataloader = self.data_configurator.train_dataloader()
        self.model.train()
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

    def _valid_one_epoch(self):
        valid_dataloader = self.data_configurator.valid_dataloader()
        self.model.eval()
        with torch.no_grad():
            for i, window in enumerate(valid_dataloader):
                window = window.to(self.device)

                anomaly_scores, centroids_diff, out_of_drnn = self.model.forward(window)

                loss = self._get_loss(anomaly_scores, centroids_diff, out_of_drnn, window)

        return loss.item()
