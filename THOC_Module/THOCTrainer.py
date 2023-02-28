from THOCBase import THOCBase
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import StepLR
import torch
import datetime

from common.utils import util_save_log_image_with_label


class THOCTrainer(THOCBase):
    def __init__(self, model_params, logger_params, run_params, optimizer_params):
        super(THOCTrainer, self).__init__(model_params, logger_params, run_params, optimizer_params)

        self.optimizer = Optimizer(self.model.parameters(), optimizer_params['lr'])
        self.scheduler = StepLR(self.optimizer, step_size=optimizer_params['decay_step'], gamma=optimizer_params['lr_decay'])

        self.loss_best = float("inf")
        self.loss_maintain = 0

    def run(self):

        print('**************** Training Start ****************')
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        for epoch in range(self.run_params['epochs']):

            if epoch == 35 :
                print(epoch)
            # Training and Validation
            train_loss = self._train_one_epoch(epoch)
            valid_loss = self._valid_one_epoch()

            self.result_log.append('train_loss', epoch, train_loss)
            self.result_log.append('valid_loss', epoch, valid_loss)

            if valid_loss < self.loss_best:
                self.loss_best = valid_loss
                self.loss_maintain = 0
                self._save_params()
            else:
                self.loss_maintain += 1

            # Logging
            self._log(epoch, self.run_params['epochs'], train_loss, valid_loss, self.loss_best, early_stop=False)
            if self.loss_maintain >= (self.run_params['max_loss_maintain']+1):
                self._log(epoch, self.run_params['epochs'], train_loss, valid_loss, self.loss_best, early_stop=True)
                break

        # Save Loss Images
        image_prefix = f'{self.result_folder}/img'
        util_save_log_image_with_label(image_prefix, self.run_params['logging']['log_image_params'],
                                       self.result_log, labels=['train_loss', 'valid_loss'])

        print('**************** Training Done ****************')

        return self.loss_best, self.model.state_dict()

    def _train_one_epoch(self, epoch):
        train_dataloader = self.data_configurator.train_dataloader()
        self.model.train()

        for i, window in enumerate(train_dataloader):
            if i != 0:
                window = window.to(self.device)

                anomaly_scores, centroids_diff, out_of_drnn = self.model.forward(window, i, epoch)

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

                anomaly_scores, centroids_diff, out_of_drnn = self.model.forward(window, i)

                loss = self._get_loss(anomaly_scores, centroids_diff, out_of_drnn, window)

        return loss.item()
