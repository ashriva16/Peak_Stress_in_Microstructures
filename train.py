import numpy as np
import torch
from utils import get_opts
import models
from data_loader import Microstructure_Data_Loader
from trainer import Trainer

SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(config):
    data_loader = Microstructure_Data_Loader('/Ankit/data/grain_data.h5')
    train_loader, test_loader = data_loader.get_data()

    model = models.autoencoder_A4_R_Nopad_32_0_Dense_Dec()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.l,
                                 weight_decay=config.w,
                                 betas=(config.b1, config.b2))

    model_trainer = Trainer(model, model.criterion, optimizer, config,
                            train_loader, config.n)

    final_train_loss = model_trainer.train_model()
    final_test_loss = model_trainer.eval_model(
        test_loader, directory="test", save_data=True)

    print("Final loss ->",
          "\t Train loss: ", final_train_loss,
          "\t Test loss: ", final_test_loss,
          "\n",
          "best_model_index: ", model_trainer.best_model_epoch,
          ", val loss: ", model_trainer.min_val_loss,
          " final epoch: ", model_trainer.last_epoch,
          " Run time: ", model_trainer.train_time,
          file=open("train.log", "a"),
          flush=True)


if __name__ == '__main__':

    config = get_opts()
    main(config)
