from collections import OrderedDict
import torch


class autoencoder_A4_R_Nopad_32_0_Dense_Dec(torch.nn.Module):
    def __init__(self):
        super(autoencoder_A4_R_Nopad_32_0_Dense_Dec, self).__init__()
        self.criterion = torch.nn.MSELoss()
        self.case = 'A4_R_Nopad_32_0_Dense_Dec'
        self.encoder = torch.nn.Sequential(
            OrderedDict([
                ('1_e_conv1', torch.nn.Conv2d(4, 64, 8, stride=8,
                                              padding=0)),  # b, 64, 16, 16
                ('2_e_act1', torch.nn.ReLU(True)),
                ('3_e_maxpool1', torch.nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
                ('4_e_conv2', torch.nn.Conv2d(64, 16, 3, stride=1,
                                              padding=0)),  # b, 16, 6, 6
                ('5_e_act2', torch.nn.ReLU(True)),
                ('6_e_maxpool2', torch.nn.MaxPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.decoder = torch.nn.Sequential(
            OrderedDict([
                ('7_d_convt1', torch.nn.ConvTranspose2d(16, 64, 2,
                                                        stride=2)),  # b, 64, 10, 10
                ('8_d_act1', torch.nn.ReLU(True)),
                ('9_d_convt2', torch.nn.ConvTranspose2d(64,
                                                        8,
                                                        3,
                                                        stride=1,
                                                        padding=0)),  # b, 8, 12, 12
                ('10_d_act2', torch.nn.ReLU(True)),
                ('11_d_convt3',
                 torch.nn.ConvTranspose2d(8, 1, 10, stride=2,
                                          padding=0)),  # b, 1, 32, 32
                ('12_out', torch.nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1
