from THOCBase import THOCBase
from THOCValidator import THOCValidator as THOCVal
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
        for epoch in range(self.run_params['epochs']):
            train_loss = self._train_one_epoch()
            valid_loss = self._valid_one_epoch()

            self.result_log.append('train_loss', epoch, train_loss)
            self.result_log.append('valid_loss', epoch, valid_loss)

            if valid_loss < self.loss_best :
                self.loss_best = valid_loss
                self.loss_maintain = 0
            else :
                self.loss_maintain += 1

            print('~~~')

            if self.loss_change >= self.run_params['max_loss_maintain'] :
                break



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
        with torch.no_grad() :
            for i, window in enumerate(valid_dataloader):
                window = window.to(self.device)

                anomaly_scores, centroids_diff, out_of_drnn = self.model.forward(window)

                loss = self._get_loss(anomaly_scores, centroids_diff, out_of_drnn, window)

        return loss.item()