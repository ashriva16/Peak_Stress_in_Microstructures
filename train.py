import sys
import numpy as np
import torch

sys.path.append('utils')

import models
from hotspots_analysis import run_postprocess

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():

    args = get_opts()
    data_loader = Microstructure_Data_Loader()
    train_loader, test_loader = data_loader.get_data()

    model = models.autoencoder_A4_R_Nopad_32_0_Dense_Dec_upscale2().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=args.l,
                                  weight_decay=args.w,
                                  betas=(args.b1, args.b2))

    trainer = Trainer(model, optimizer, device,
                      train_loader, val_loader,
                      args.n, directory)

    final_train_loss = trainer.train_model()
    final_test_loss = trainer.eval_model(test_loader)

    print("Final loss ->",
        "\t Train loss: ", final_train_loss,
        "\t Test loss: ", final_test_loss,
        "\n",
        "best_model_index: ", trainer.best_model_epoch,
        ", val loss: ", trainer.min_val_loss,
        " final epoch: ", trainer.epoch,
        " Run time: ", trainer.train_time,
        file=open(filetxt, "a"),
        flush=True)