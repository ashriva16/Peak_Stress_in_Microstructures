#!/usr/bin/python3
import os
import time
from copy import deepcopy
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
import torch
import h5py
import numpy as np
import utils
from utils import Logger


class Trainer():
    """
    Trainer class
    TODO : Hyperparameter optimization, K-fold Crossvalidation, MultiGPU capability
    """

    def __init__(self, model, metric_ftns, optimizer, config,
                 train_loader, len_epochs, save_directory="results/",
                 job="0", lr_scheduler=None):

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_loader = train_loader
        self.model = model.to(self.device)
        self.best_model = deepcopy(model).to(self.device)
        self.optimizer = optimizer
        self.criterion = metric_ftns
        self.total_epoch = len_epochs

        self.output_dir = save_directory + model.case + '/train_' + job + "/"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.val_calc_epoch = 10
        self.checkpoint_epoch = 100
        self.batch_size = 1000
        self.scheduler = lr_scheduler
        self.max_patience = 200

        self.train_loss_arr = []
        self.val_loss_arr = []
        self.train_epoch = []
        self.val_epoch = []
        self.min_val_loss = float("inf")

        self.val_loader = None
        self.train_loader = None
        self.epoch_start = None
        self.train_loss = None
        self.train_time = None
        self.last_epoch = None
        self.best_model_epoch = None
        self.patience = None

    def _split_train_val_data(self):

        total_number_of_samples = len(self.data_loader)
        number_of_val_samples = int(.1 * total_number_of_samples)
        number_of_training_samples = total_number_of_samples - number_of_val_samples
        train_loader, val_loader = torch.utils.data.random_split(self.data_loader,
                                                                 [number_of_training_samples,
                                                                  number_of_val_samples]
                                                                 )

        self.train_loader = data_utils.DataLoader(train_loader,
                                                  batch_size=self.batch_size,
                                                  shuffle=False)

        self.val_loader = data_utils.DataLoader(val_loader,
                                                batch_size=self.batch_size,
                                                shuffle=False)

    def _train_epoch(self, epoch):

        # Minibatch gradient descent
        self.model.train()
        train_loss = 0
        for data in self.train_loader:
            X, Y = data
            X = X.to(self.device)
            Y = Y.to(self.device)
            # ===================forward=====================
            pred, _ = self.model(X)
            loss = self.criterion(pred, Y)
            # ===================backward====================
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # # ===================train error=================
            if(epoch % self.val_calc_epoch == 0):
                self.model.eval()
                with torch.no_grad():
                    pred, _ = self.model(X)
                    train_loss += self.criterion(pred.detach(),
                                                 Y.detach()).item()
                self.model.train()

        if(epoch % self.val_calc_epoch == 0):
            train_loss = train_loss / len(self.train_loader)
            self.train_loss_arr.append(train_loss)
            self.train_epoch.append(epoch)

        return train_loss

    def _valid_epoch(self, epoch):

        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in self.val_loader:
                X, Y = data
                X = X.to(self.device)
                Y = Y.to(self.device)
                pred, _ = self.model(X)
                val_loss += self.criterion(pred.detach(),
                                           Y.detach()).item()

        val_loss = val_loss / len(self.val_loader)
        self.val_loss_arr.append(val_loss)
        self.val_epoch.append(epoch)

        if self.val_loss_arr[-1] <= self.min_val_loss:
            self.best_model.load_state_dict(self.model.state_dict())
            self.best_model_epoch = epoch
            self.min_val_loss = self.val_loss_arr[-1]
            self._save_checkpoint(epoch, save_best=True)

        return val_loss

    def _save_checkpoint(self, epoch, save_best=False):

        if save_best:
            # Save the best model
            checkpoint = {'epoch': self.best_model_epoch,
                          'model_state_dict': self.best_model.state_dict(),
                          'loss': self.min_val_loss
                          }
            torch.save(checkpoint, self.output_dir + 'best_trainedmodel.th')
        else:
            # Save the model checkpoint
            checkpoint = {'epoch': epoch,
                          'model_state_dict': self.model.state_dict(),
                          'optimizer1': self.optimizer.state_dict(),
                          'loss': self.val_loss_arr[-1]
                          }

            torch.save(checkpoint, self.output_dir +
                       self.model.case + "_" + str(epoch) +
                       '.th')

    def _logger(self, epoch):
        # ================================================================== #
        #                        Tensorboard Logging                         #
        # ================================================================== #

        logger_train = Logger(
            self.output_dir + 'logs_' + "/train")
        logger_val = Logger(self.output_dir + 'logs_' + "/val")

        logger_train.scalar_summary(
            'aloss/loss', self.train_loss_arr[-1], epoch)
        logger_val.scalar_summary(
            'aloss/loss', self.val_loss_arr[-1], epoch)

        # Not helping so no point of plotting them, requires too much memory
        # for tag, value in model.named_parameters():
        # 	tag = tag.replace('.', '/')
        # 	logger.layer_summary(tag, value.detach().cpu(), epoch)
        # 	logger.histo_summary(tag, value.detach().cpu().numpy(), epoch)
        # 	# logger.histo_summary(tag + '/grad',
        # 	#                      value.grad.data.cpu().numpy(), epoch)

    def _plot(self):
        np.savez(self.output_dir + '/error_hist.npz',
                 train_epoch=self.train_epoch, train_loss_arr=self.train_loss_arr,
                 val_epoch=self.val_epoch, val_loss_arr=self.val_loss_arr)

        plt.switch_backend('agg')
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(self.train_epoch,
                self.train_loss_arr, 'b', label='training loss')
        plt.plot(self.val_epoch,
                 self.val_loss_arr, 'r', label='val loss')
        plt.title('Training and Validation loss')
        plt.xlabel('iterations ', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.yscale('log')
        ax.legend()
        filename = self.output_dir + '/loss'
        fig.savefig(filename, transparent=True)

    def crossvalidate_model(self, fold):
        pass

    def _earlystopping(self, val_loss):

        if(val_loss <= self.min_val_loss):
            self.patience = 0
        else:
            self.patience += 1

        if(self.patience > self.max_patience):
            return True

    def train_model(self):

        start = time.time()

        self._split_train_val_data()

        for epoch in range(1, self.total_epoch + 1):

            train_loss = self._train_epoch(epoch)

            # validation error
            if epoch % self.val_calc_epoch == 0:
                valid_loss = self._valid_epoch(epoch)
                print("Epoch: ", epoch,
                      " Train loss: ", train_loss,
                      " Valid loss: ", valid_loss,
                      file=open("train.log", "a"),
                      flush=True)

                if self._earlystopping(valid_loss):
                    break

            # Cheakpoints to save the model #
            if epoch % self.checkpoint_epoch == 0:
                self._save_checkpoint(epoch)
                self._logger(epoch)

            if(self.scheduler is not None):
                self.scheduler.step()

        end = time.time()
        torch.cuda.empty_cache()
        self.train_time = end - start
        self.last_epoch = epoch
        self._plot()

        # Saving model for Inference
        # it is only necessary to save the trained model’s learned parameters.
        # Saving the model’s state_dict with the torch.save()
        # function will give you the most flexibility for
        # restoring the model later, which is why
        # it is the recommended method for saving models.
        self.train_loss = self.eval_model(
            self.data_loader, directory="train", save_data=True)
        return self.train_loss

    def eval_model(self, data_loader, directory="", save_data=False):

        data_loader = data_utils.DataLoader(data_loader,
                                            batch_size=1000,
                                            shuffle=False)

        loss = 0
        n = len(data_loader) * 1000
        count = 1000
        with torch.no_grad():
            for data in data_loader:
                X, Y = data
                X = X.to(self.device)
                Y = Y.to(self.device)
                output, _ = self.model(X)
                loss = +self.criterion(output.data, Y.data).item()

                if(save_data):
                    with h5py.File(self.output_dir + directory + '_pred.h5', 'a') as f:
                        utils.store_data_in_hdffile("x",
                                                    X.cpu().transpose_(3, 1).data.numpy(),
                                                    f, count, count + 1000,
                                                    n)

                        utils.store_data_in_hdffile("y",
                                                    Y.cpu().transpose_(3, 1).data.numpy(),
                                                    f, count, count + 1000,
                                                    n)

                        utils.store_data_in_hdffile("pred",
                                                    output.cpu().transpose_(3, 1).data.numpy(),
                                                    f, count, count + 1000,
                                                    n)

                    count += 1000

        loss = loss / len(data_loader)
        return loss
