#!/usr/bin/python3
import os
import time
from logger import Logger
import matplotlib.pyplot as plt
import torch.utils.data as data_utils

class Trainer():
    """
    Trainer class
    """
    def __init__(self, model, metric_ftns, optimizer, config, device,
                 train_loader, len_epochs, save_directory="results/", job=None, lr_scheduler=None):

    self.data_loader = train_loader
    self.val_loader = val_loader
    self.model = model
    self.best_model = deepcopy(model)
    self.optimizer = optimizer
    self.criterion = model.criterion
    self.output_dir = save_directory + model.case + '/train_' + job + "/"

    if not os.path.exists(self.output_dir):
        os.makedirs(self.output_dir)

    self.train_loss_arr = []
    self.val_loss_arr = []
    self.min_val_loss = float("inf")
    self.total_epoch = len_epochs
    
    def _train_epoch(self, epoch):
        
        # Minibatch gradient descent
        self.model.train()
        train_loss = 0
        for data in train_loader:
            X, Y = data
            X = X.to(device)
            Y = Y.to(device)
            # ===================forward=====================
            pred, mid = self.model(X)
            loss = self.criterion(pred, Y)
            # ===================backward====================
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # # ===================train error=================
            if(epoch % self.val_epoch == 0):
                self.model.eval()
                pred, mid = self.model(X)
                train_loss += self.criterion(pred.detach(), Y.detach()).item()

        train_loss = train_loss / len(self.train_loader)
        self.train_loss_arr.append(train_loss)

        scheduler.step()

    def _valid_epoch(self, epoch):

        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                X, Y = data
                X = X.to(device)
                Y = Y.to(device)
                pred, mid = self.model(X)
                val_loss += self.criterion(pred.detach(), Y.detach()).item()

        val_loss = val_loss / len(val_loader)
        self.val_loss_arr.append(val_loss)

    def _save_checkpoint(self, epoch, save_best=False):

        if(save_best):
            checkpoint = {'epoch': self.best_model_epoch,
                        'model_state_dict': self.best_model.state_dict(),
                        'loss': self.min_val_loss
                        }
            torch.save(checkpoint, self.output_dir + 'best_trainedmodel' +
                    "_" + str(epoch_start) + "_" + str(args.n - 1) +
                    '.th')

        else:
            checkpoint = {'epoch': epoch + epoch_start,
                'model_state_dict': model.state_dict(),
                'optimizer1': optimizer.state_dict()
                'loss': loss
                }

            torch.save(checkpoint, self.output_dir +
                    self.model.case + "_" + args.job +
                    "_" + str(epoch_start) + "_" + str(args.n - 1) +
                    '.th')

    def _logger(self,epoch):
            # ================================================================== #
            #                        Tensorboard Logging                         #
            # ================================================================== #

            logger_train = Logger(self.output_dir + 'logs_' + str(args.c) + "/train")
            logger_val = Logger(self.output_dir + 'logs_' + str(args.c) + "/val")

            logger_train.scalar_summary('aloss/loss', train_loss, epoch + 1)
            logger_val.scalar_summary('aloss/loss', val_loss, epoch + 1)

            # Not helping so no point of plotting them, requires too much memory
            # for tag, value in model.named_parameters():
            # 	tag = tag.replace('.', '/')
            # 	logger.layer_summary(tag, value.detach().cpu(), epoch + 1)
            # 	logger.histo_summary(tag, value.detach().cpu().numpy(), epoch + 1)
            # 	# logger.histo_summary(tag + '/grad',
            # 	#                      value.grad.data.cpu().numpy(), epoch + 1)

    def _plot():
        plt.switch_backend('agg')
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(range(len(train_loss_arr)), train_loss_arr, 'b', label='training loss')
        plt.plot(range(len(val_loss_arr)), val_loss_arr, 'r', label='val loss')
        plt.title('Training and Test loss')
        plt.xlabel('iterations ', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.yscale('log')
        ax.legend()
        filename = self.output_dir + '/loss'
        fig.savefig(filename, transparent=True)

    def _split_train_val_data(self):
        
        TOTAL_NUMBER_OF_SAMPLES = len(self.data_loader)
        NUMBER_OF_VAL_SAMPLES = int(.1 * TOTAL_NUMBER_OF_SAMPLES)
        NUMBER_OF_TRAINING_SAMPLES = TOTAL_NUMBER_OF_SAMPLES - NUMBER_OF_VAL_SAMPLES
        train_loader, val_loader = torch.utils.data.random_split( self.data_loader, 
                                                                [NUMBER_OF_TRAINING_SAMPLES,
                                                                NUMBER_OF_VAL_SAMPLES]
                                                                )

        self.train_loader = data_utils.DataLoader(train_loader,
                                            batch_size=batch_size,
                                            shuffle=False)

        self.val_loader = data_utils.DataLoader(val_loader,
                                        batch_size=batch_size,
                                        shuffle=False)

    def train_model():

        start = time.time()
        min_val_loss = float("inf")
        patience = 0

        for epoch in range(self.total_epoch):
            
            self._train_epoch(epoch)
            # validation error 
            if(epoch % val_epoch == 0):
                self._valid_epoch(epoch)

            # Cheakpoints to save the model #
            if(epoch % checkpoint_epoch == 0):
                self._save_checkpoint(epoch)
                self._logger(epoch)

            if (val_loss <= self.min_val_loss):
                self.best_model.load_state_dict(self.model.state_dict())
                self.best_model_epoch = epoch
                self.min_val_loss = self.val_loss_arr[-1]
                
                self._save_checkpoint(epoch,save_best=True)

                patience = 0
            else:
                patience += 1

            # ================== Stop training ================= #
            if(patience > 200):
                break;

        self._plot()
        torch.cuda.empty_cache()

        end = time.time()
        self.train_time = end - start
        self.epoch = epoch

        # Saving model for Inference
        # it is only necessary to save the trained model’s learned parameters.
        # Saving the model’s state_dict with the torch.save()
        # function will give you the most flexibility for
        # restoring the model later, which is why
        # it is the recommended method for saving models.
        self._save_checkpoint(self.best_model_epoch, save_best=True)
        train_loss = self.eval_model(self.data_loader)
        return train_loss

    def eval_model(self, data_loader, directory="", save_data=False):
        
        data_loader = data_utils.DataLoader(data_loader,
                                            batch_size=1000,
                                            shuffle=False)

        loss = 0
        n = len(data_loader)*1000
        count = 1000
        with torch.no_grad():
            for data in data_loader:
                X, Y = data
                X = X.to(device)
                Y = Y.to(device)
                output, mid = self.model(X)
                loss = +self.criterion(output.data, Y.data).item()

                if(save_data):
                    with h5py.File(self.output_dir+directory+'/pred.h5', 'a') as f:
                        store_data_in_hdffile("x",
                                                X.cpu().transpose_(3, 1).data.numpy(),
                                                f, count, count + 1000,
                                                n)

                        store_data_in_hdffile("y",
                                                Y.cpu().transpose_(3, 1).data.numpy(),
                                                f, count, count + 1000,
                                                n)

                        store_data_in_hdffile("pred",
                                               output.cpu().transpose_(3, 1).data.numpy(),
                                               f, count, count + 1000,
                                               n)
                    
                    count += 1000

        loss = loss / len(data_loader)
        return loss

