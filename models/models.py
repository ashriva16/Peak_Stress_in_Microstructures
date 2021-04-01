import torch
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F
from torch.nn.modules.upsampling import Upsample
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Flatten(nn.Module):
    def forward(self, _input_):
        return _input_.view(_input_.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, _input_, sizec=16, sizex=2, sizey=2):
        return _input_.view(_input_.size(0), sizec, sizex, sizey)


class Cosine(nn.Module):
    def forward(self, input1, input2):
        cos = nn.CosineSimilarity(eps=1e-6)
        dot = cos(input1.view(input1.size(0), -1),
                  input2.view(input2.size(0), -1))
        return torch.mean(-dot)
# Default autoencoder_regression_Default


class scale01(nn.Module):
    def forward(self, image):

        temp_sub = torch.zeros(image.shape)
        temp_div = torch.zeros(image.shape)
        for i in range(image.shape[1]):
            vector = image[0, i, :, :].detach()
            min_v = torch.min(vector).item()
            max_v = torch.max(vector).item()
            if(max_v == min_v):
                temp_sub[0, i, :, :] = 0
                temp_div[0, i, :, :] = 1
            else:
                temp_sub[0, i, :, :] = min_v
                temp_div[0, i, :, :] = max_v - min_v

        image = (image - temp_sub) / temp_div

        return image


def edge_mask(x_):
    if(torch.is_tensor(x_)):
        x = x_.clone()
    else:
        x = np.copy(x_)
    x[2 * (slice(1, -1),)] = False
    return x


def left_top_edge_mask(x_):
    if(torch.is_tensor(x_)):
        x = x_.clone()
    else:
        x = np.copy(x_)
    x[2 * (slice(1, -1),)] = False
    x[:, -1] = False
    x[-1, :] = False
    return x


def right_botom_edge_mask(x_):
    if(torch.is_tensor(x_)):
        x = x_.clone()
    else:
        x = np.copy(x_)
    x[2 * (slice(1, -1),)] = False
    x[:, 0] = False
    x[0, :] = False
    return x


def non_edge_mask(x_):
    if(torch.is_tensor(x_)):
        x = x_.clone()
    else:
        x = np.copy(x_)
    x[0, slice(0, -1)] = False
    x[0, -1] = False
    x[-1, slice(0, -1)] = False
    x[-1, -1] = False
    x[slice(0, -1), 0] = False
    x[-1, 0] = False
    x[slice(0, -1), -1] = False
    x[-1, -1] = False
    return x


class RGBgradients(nn.Module):
    def __init__(self, weight):  # weight is a numpy array
        super().__init__()
        k_height, k_width = weight.shape[1:]
        # assuming that the height and width of the kernel are always odd numbers
        padding_x = int((k_height - 1) / 2)
        padding_y = int((k_width - 1) / 2)

        # convolutional layer with 3 in_channels and 6 out_channels
        # the 3 in_channels are the color channels of the image
        # for each in_channel we have 2 out_channels corresponding to the x and the y gradients
        self.conv = nn.Conv2d(1, 2, (k_height, k_width), bias=False,
                              padding=(padding_x, padding_y))

        # initialize the weights of the convolutional layer to be the one provided
        # the weights correspond to the x/y filter for the channel in question and zeros for other channels
        weight1x = np.array([weight[0]])  # x-derivative for 1st in_channel

        weight1y = np.array([weight[1]])  # y-derivative for 1st in_channel

        if(device.type == "cuda"):
            weight_final = torch.from_numpy(
                np.array([weight1x, weight1y])).type(torch.cuda.FloatTensor)
        else:
            weight_final = torch.from_numpy(
                np.array([weight1x, weight1y])).type(torch.cuda.FloatTensor)

        if self.conv.weight.shape == weight_final.shape:
            self.conv.weight = nn.Parameter(weight_final)
            self.conv.weight.requires_grad_(False)
        else:
            print('Error: The shape of the given weights is not correct')

    # Note that a second way to define the conv. layer here would be to pass group = 3 when calling torch.nn.Conv2d
    def forward(self, x):
        return self.conv(x)


def get_gradient(img):
    filter_x_gb = np.array([[0, 0, 0],
                            [-1, 0, 1],
                            [0, 0, 0]])
    filter_y_gb = filter_x_gb.T
    grad_filters_gb = np.array([filter_x_gb, filter_y_gb])
    gradLayer_gb = RGBgradients(grad_filters_gb)
    img_grad = gradLayer_gb(img)

    temp_x = (img_grad[:, 0, :, :]**2)
    temp_y = (img_grad[:, 1, :, :]**2)

    grad_img = temp_x + temp_y

    return grad_img
    # return non_edge_mask(grad_img) + right_botom_edge_mask(grad_img) + left_top_edge_mask(grad_img)


class autoencoder_A1_Default32_R_0_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_A1_Default32_R_0_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'A1_Default32_R_0_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 64, 13, stride=13,
                                        padding=1)),  # b, 64, 10, 10
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 5, 5
                ('4_e_conv2', nn.Conv2d(64, 16, 3, stride=2,
                                        padding=1)),  # b, 16, 2, 2
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16*2*2
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(16, 64, 2,
                                                  stride=2)),  # b, 64, 4, 4
                ('8_d_act1', nn.ReLU(True)),
                ('9_d_convt2', nn.ConvTranspose2d(64,
                                                  8,
                                                  8,
                                                  stride=2,
                                                  padding=1)),  # b, 8, 12, 12
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(8, 1, 12, stride=2,
                                    padding=1)),  # b, 1, 32, 32
                ('12_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_A2_R_Nopad_32_0_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_A2_R_Nopad_32_0_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'A2_R_Nopad_32_0_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 64, 9, stride=7,
                                        padding=0)),  # b, 64, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
                ('4_e_conv2', nn.Conv2d(64, 16, 3, stride=2,
                                        padding=0)),  # b, 16, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(16, 64, 2,
                                                  stride=2)),  # b, 64, 10, 10
                ('8_d_act1', nn.ReLU(True)),
                ('9_d_convt2', nn.ConvTranspose2d(64,
                                                  8,
                                                  4,
                                                  stride=2,
                                                  padding=0)),  # b, 8, 12, 12
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(8, 1, 6, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('12_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_A3_R_Nopad_32_0_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_A3_R_Nopad_32_0_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'A3_R_Nopad_32_0_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 64, 5, stride=3,
                                        padding=0)),  # b, 64, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
                ('4_e_conv2', nn.Conv2d(64, 16, 3, stride=2,
                                        padding=0)),  # b, 16, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(16, 64, 2,
                                                  stride=1)),  # b, 64, 10, 10
                ('8_d_act1', nn.ReLU(True)),
                ('9_d_convt2', nn.ConvTranspose2d(64,
                                                  8,
                                                  5,
                                                  stride=1,
                                                  padding=0)),  # b, 8, 12, 12
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(8, 1, 6, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('12_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_A4_R_Nopad_32_0_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_A4_R_Nopad_32_0_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'A4_R_Nopad_32_0_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 64, 8, stride=8,
                                        padding=0)),  # b, 64, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
                ('4_e_conv2', nn.Conv2d(64, 16, 3, stride=1,
                                        padding=0)),  # b, 16, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(16, 64, 2,
                                                  stride=2)),  # b, 64, 10, 10
                ('8_d_act1', nn.ReLU(True)),
                ('9_d_convt2', nn.ConvTranspose2d(64,
                                                  8,
                                                  3,
                                                  stride=1,
                                                  padding=0)),  # b, 8, 12, 12
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(8, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('12_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_A4_R_Nopad_32_0_Dense_Dec_upscale(nn.Module):
    def __init__(self):
        super(autoencoder_A4_R_Nopad_32_0_Dense_Dec_upscale, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'A4_R_Nopad_32_0_Dense_Dec_upscale'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 64, 8, stride=8,
                                        padding=0)),  # b, 64, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
                ('4_e_conv2', nn.Conv2d(64, 16, 3, stride=1,
                                        padding=0)),  # b, 16, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(16, 64, 2,
                                                  stride=2)),  # b, 64, 10, 10
                ('8_d_act1', nn.ReLU(True)),
                ('9_d_convt2', nn.ConvTranspose2d(64,
                                                  8,
                                                  3,
                                                  stride=1,
                                                  padding=0)),  # b, 8, 12, 12
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(8, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('12_out', nn.Tanh()),
                ('13_d_upscale',
                 Upsample(scale_factor=4, mode='bilinear', align_corners=True))
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_A4_R_Nopad_32_0_Dense_Dec_upscale2(nn.Module):
    def __init__(self):
        super(autoencoder_A4_R_Nopad_32_0_Dense_Dec_upscale2, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'A4_R_Nopad_32_0_Dense_Dec_upscale2'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 64, 8, stride=8,
                                        padding=0)),  # b, 64, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
                ('4_e_conv2', nn.Conv2d(64, 16, 3, stride=1,
                                        padding=0)),  # b, 16, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(16, 64, 2,
                                                  stride=2)),  # b, 64, 10, 10
                ('8_d_act1', nn.ReLU(True)),
                ('9_d_convt2', nn.ConvTranspose2d(64,
                                                  8,
                                                  3,
                                                  stride=1,
                                                  padding=0)),  # b, 8, 12, 12
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(8, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('12_d_upscale',
                 Upsample(scale_factor=4, mode='bilinear', align_corners=True)),
                ('13_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_A4_R_Nopad_32_1_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_A4_R_Nopad_32_1_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'A4_R_Nopad_32_1_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 16, 8, stride=8,
                                        padding=0)),  # b, 16, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 16, 8, 8
                ('4_e_conv2', nn.Conv2d(16, 8, 3, stride=1,
                                        padding=0)),  # b, 8, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 8, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(8, 16, 2,
                                                  stride=2)),  # b, 16, 10, 10
                ('8_d_act1', nn.ReLU(True)),
                ('9_d_convt2', nn.ConvTranspose2d(16,
                                                  4,
                                                  3,
                                                  stride=1,
                                                  padding=0)),  # b, 4, 12, 12
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(4, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('12_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_A4_R_Nopad_32_2_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_A4_R_Nopad_32_2_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'A4_R_Nopad_32_2_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 128, 8, stride=8,
                                        padding=0)),  # b, 16, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 16, 8, 8
                ('4_e_conv2', nn.Conv2d(128, 32, 3, stride=1,
                                        padding=0)),  # b, 8, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 8, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(32, 128, 2,
                                                  stride=2)),  # b, 16, 10, 10
                ('8_d_act1', nn.ReLU(True)),
                ('9_d_convt2', nn.ConvTranspose2d(128,
                                                  16,
                                                  3,
                                                  stride=1,
                                                  padding=0)),  # b, 4, 12, 12
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(16, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('12_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_A4_R_Nopad_32_3_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_A4_R_Nopad_32_3_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'A4_R_Nopad_32_3_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 256, 8, stride=8,
                                        padding=0)),  # b, 16, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 16, 8, 8
                ('4_e_conv2', nn.Conv2d(256, 64, 3, stride=1,
                                        padding=0)),  # b, 8, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 8, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(64, 256, 2,
                                                  stride=2)),  # b, 16, 10, 10
                ('8_d_act1', nn.ReLU(True)),
                ('9_d_convt2', nn.ConvTranspose2d(256,
                                                  32,
                                                  3,
                                                  stride=1,
                                                  padding=0)),  # b, 4, 12, 12
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(32, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('12_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_A4_R_Nopad_32_4_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_A4_R_Nopad_32_4_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'A4_R_Nopad_32_4_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 512, 8, stride=8,
                                        padding=0)),  # b, 16, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 16, 8, 8
                ('4_e_conv2', nn.Conv2d(512, 128, 3, stride=1,
                                        padding=0)),  # b, 8, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 8, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(128, 512, 2,
                                                  stride=2)),  # b, 16, 10, 10
                ('8_d_act1', nn.ReLU(True)),
                ('9_d_convt2', nn.ConvTranspose2d(512,
                                                  64,
                                                  3,
                                                  stride=1,
                                                  padding=0)),  # b, 4, 12, 12
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(64, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('12_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_A4_R_Nopad_32_5_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_A4_R_Nopad_32_5_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'A4_R_Nopad_32_5_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 64, 8, stride=8,
                                        padding=0)),  # b, 16, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 16, 8, 8
                ('4_e_conv2', nn.Conv2d(64, 16, 3, stride=1,
                                        padding=0)),  # b, 8, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 8, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(16, 8, 2,
                                                  stride=2)),  # b, 16, 10, 10
                ('8_d_act1', nn.ReLU(True)),
                ('9_d_convt2', nn.ConvTranspose2d(8,
                                                  4,
                                                  3,
                                                  stride=1,
                                                  padding=0)),  # b, 4, 12, 12
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(4, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('12_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_A9_R_Nopad_32_0_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_A9_R_Nopad_32_0_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'A9_R_Nopad_32_0_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 64, 8, stride=8,
                                        padding=0)),  # b, 16, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 16, 8, 8
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('4_d_convt1', nn.ConvTranspose2d(64, 32, 2,
                                                  stride=1)),  # b, 32, 9, 9
                ('5_d_act1', nn.ReLU(True)),
                ('6_d_convt2', nn.ConvTranspose2d(32,
                                                  8,
                                                  4,
                                                  stride=1,
                                                  padding=0)),  # b, 4, 12, 12
                ('7_d_act2', nn.ReLU(True)),
                ('8_d_convt3',
                 nn.ConvTranspose2d(8, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('9_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_A10_R_Nopad_32_0_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_A10_R_Nopad_32_0_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'A10_R_Nopad_32_0_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 64, 8, stride=8,
                                        padding=0)),  # b, 64, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_conv2', nn.Conv2d(64, 64, 2, stride=2, padding=0)),  # b, 64, 8, 8
                ('4_e_act2', nn.ReLU(True)),
                ('5_e_conv3', nn.Conv2d(64, 16, 3, stride=1,
                                        padding=0)),  # b, 16, 6, 6
                ('6_e_act3', nn.ReLU(True)),
                ('7_e_maxpool1', nn.MaxPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('8_d_convt1', nn.ConvTranspose2d(16, 64, 2,
                                                  stride=2)),  # b, 64, 10, 10
                ('9_d_act1', nn.ReLU(True)),
                ('10_d_convt2', nn.ConvTranspose2d(64,
                                                   8,
                                                   3,
                                                   stride=1,
                                                   padding=0)),  # b, 8, 12, 12
                ('11_d_act2', nn.ReLU(True)),
                ('12_d_convt3',
                 nn.ConvTranspose2d(8, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('13_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_A4_R_Nopad_32_0_Dense_Inc(nn.Module):
    def __init__(self):
        super(autoencoder_A4_R_Nopad_32_0_Dense_Inc, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'A4_R_Nopad_32_0_Dense_Inc'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 16, 8, stride=8,
                                        padding=0)),  # b, 16, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 16, 8, 8
                ('4_e_conv2', nn.Conv2d(16, 64, 3, stride=1,
                                        padding=0)),  # b, 64, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 64, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(64, 8, 2,
                                                  stride=2)),  # b, 8, 10, 10
                ('8_d_act1', nn.ReLU(True)),
                ('9_d_convt2', nn.ConvTranspose2d(8,
                                                  4,
                                                  3,
                                                  stride=1,
                                                  padding=0)),  # b, 4, 12, 12
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(4, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('12_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_A4_R_Nopad_32_1_Dense_Inc(nn.Module):
    def __init__(self):
        super(autoencoder_A4_R_Nopad_32_1_Dense_Inc, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'A4_R_Nopad_32_1_Dense_Inc'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 16, 8, stride=8,
                                        padding=0)),  # b, 16, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 16, 8, 8
                ('4_e_conv2', nn.Conv2d(16, 64, 3, stride=1,
                                        padding=0)),  # b, 64, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 64, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(64, 32, 2,
                                                  stride=2)),  # b, 8, 10, 10
                ('8_d_act1', nn.ReLU(True)),
                ('9_d_convt2', nn.ConvTranspose2d(32,
                                                  16,
                                                  3,
                                                  stride=1,
                                                  padding=0)),  # b, 4, 12, 12
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(16, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('12_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_A4_R_Nopad_32_0_Dense_Dec_sigmoid(nn.Module):
    def __init__(self):
        super(autoencoder_A4_R_Nopad_32_0_Dense_Dec_sigmoid, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'A4_R_Nopad_32_0_Dense_Dec_sigmoid'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 64, 8, stride=8,
                                        padding=0)),  # b, 64, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
                ('4_e_conv2', nn.Conv2d(64, 16, 3, stride=1,
                                        padding=0)),  # b, 16, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(16, 64, 2,
                                                  stride=2)),  # b, 64, 10, 10
                ('8_d_act1', nn.ReLU(True)),
                ('9_d_convt2', nn.ConvTranspose2d(64,
                                                  8,
                                                  3,
                                                  stride=1,
                                                  padding=0)),  # b, 8, 12, 12
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(8, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('12_out', nn.Sigmoid())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_A4_R_Nopad_32_0_Dense_Dec_cosine(nn.Module):
    def __init__(self):
        super(autoencoder_A4_R_Nopad_32_0_Dense_Dec_cosine, self).__init__()
        self.criterion = cosine()
        self.case = 'A4_R_Nopad_32_0_Dense_Dec_cosine'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 64, 8, stride=8,
                                        padding=0)),  # b, 64, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
                ('4_e_conv2', nn.Conv2d(64, 16, 3, stride=1,
                                        padding=0)),  # b, 16, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(16, 64, 2,
                                                  stride=2)),  # b, 64, 10, 10
                ('8_d_act1', nn.ReLU(True)),
                ('9_d_convt2', nn.ConvTranspose2d(64,
                                                  8,
                                                  3,
                                                  stride=1,
                                                  padding=0)),  # b, 8, 12, 12
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(8, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('12_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_A4_R_Nopad_32_0_Dense_Dec_avgpool(nn.Module):
    def __init__(self):
        super(autoencoder_A4_R_Nopad_32_0_Dense_Dec_avgpool, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'A4_R_Nopad_32_0_Dense_Dec_avgpool'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 64, 8, stride=8,
                                        padding=0)),  # b, 64, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_Avgpool1', nn.AvgPool2d(2, stride=2)),  # b, 64, 8, 8
                ('4_e_conv2', nn.Conv2d(64, 16, 3, stride=1,
                                        padding=0)),  # b, 16, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_Avgpool2', nn.AvgPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(16, 64, 2,
                                                  stride=2)),  # b, 64, 10, 10
                ('8_d_act1', nn.ReLU(True)),
                ('9_d_convt2', nn.ConvTranspose2d(64,
                                                  8,
                                                  3,
                                                  stride=1,
                                                  padding=0)),  # b, 8, 12, 12
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(8, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('12_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_A4_R_Nopad_32_0_Dense_Dec_lrelu(nn.Module):
    def __init__(self):
        super(autoencoder_A4_R_Nopad_32_0_Dense_Dec_lrelu, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'A4_R_Nopad_32_0_Dense_Dec_lrelu'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 64, 8, stride=8,
                                        padding=0)),  # b, 64, 16, 16
                ('2_e_act1', nn.LeakyReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
                ('4_e_conv2', nn.Conv2d(64, 16, 3, stride=1,
                                        padding=0)),  # b, 16, 6, 6
                ('5_e_act2', nn.LeakyReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(16, 64, 2,
                                                  stride=2)),  # b, 64, 10, 10
                ('8_d_act1', nn.LeakyReLU(True)),
                ('9_d_convt2', nn.ConvTranspose2d(64,
                                                  8,
                                                  3,
                                                  stride=1,
                                                  padding=0)),  # b, 8, 12, 12
                ('10_d_act2', nn.LeakyReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(8, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('12_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_A4_R_Nopad_32_0_Dense_Dec_nogb(nn.Module):
    def __init__(self):
        super(autoencoder_A4_R_Nopad_32_0_Dense_Dec_nogb, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'A4_R_Nopad_32_0_Dense_Dec_nogb'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(3, 64, 8, stride=8,
                                        padding=0)),  # b, 64, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
                ('4_e_conv2', nn.Conv2d(64, 16, 3, stride=1,
                                        padding=0)),  # b, 16, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(16, 64, 2,
                                                  stride=2)),  # b, 64, 10, 10
                ('8_d_act1', nn.ReLU(True)),
                ('9_d_convt2', nn.ConvTranspose2d(64,
                                                  8,
                                                  3,
                                                  stride=1,
                                                  padding=0)),  # b, 8, 12, 12
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(8, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('12_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_A4_R_Nopad_32_0_Dense_Dec_BN(nn.Module):
    def __init__(self):
        super(autoencoder_A4_R_Nopad_32_0_Dense_Dec_BN, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'A4_Nopad_32_R_0_Dense_Dec_BN'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 64, 8, stride=8,
                                        padding=0)),  # b, 64, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('2_e_batchnorm', nn.BatchNorm2d(64)),
                # -----------------------------------------------------------------------------
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
                ('4_e_conv2', nn.Conv2d(64, 16, 3, stride=1,
                                        padding=0)),  # b, 16, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('5_e_batchnorm', nn.BatchNorm2d(16)),
                # -----------------------------------------------------------------------------
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(16, 64, 2,
                                                  stride=2)),  # b, 64, 10, 10
                ('8_d_act1', nn.ReLU(True)),
                ('8_d_batchnorm', nn.BatchNorm2d(64)),
                # -----------------------------------------------------------------------------
                ('9_d_convt2', nn.ConvTranspose2d(64,
                                                  8,
                                                  3,
                                                  stride=1,
                                                  padding=0)),  # b, 8, 12, 12
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(8, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('12_out', nn.Tanh())
                # -----------------------------------------------------------------------------
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_A4_R_Nopad_32_0_Dense_Dec_No_Bias(nn.Module):
    def __init__(self):
        super(autoencoder_A4_R_Nopad_32_0_Dense_Dec_No_Bias, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'A4_R_Nopad_32_0_Dense_Dec_No_Bias'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 64, 8, stride=8,
                                        padding=0, bias=False)),  # b, 64, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
                ('4_e_conv2', nn.Conv2d(64, 16, 3, stride=1,
                                        padding=0, bias=False)),  # b, 16, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(16, 64, 2,
                                                  stride=2, bias=False)),  # b, 64, 10, 10
                ('8_d_act1', nn.ReLU(True)),
                ('9_d_convt2', nn.ConvTranspose2d(64,
                                                  8,
                                                  3,
                                                  stride=1,
                                                  padding=0, bias=False)),  # b, 8, 12, 12
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(8, 1, 10, stride=2,
                                    padding=0, bias=False)),  # b, 1, 32, 32
                ('12_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1

# Old name autoencoder_A4_BN_Nopad_32_R_0_Dense_Dec_No_Bias


class autoencoder_A4_R_Nopad_32_0_Dense_Dec_BN_No_Bias(nn.Module):
    def __init__(self):
        super(autoencoder_A4_R_Nopad_32_0_Dense_Dec_BN_No_Bias, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'A4_R_Nopad_32_0_Dense_Dec_BN_No_Bias'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 64, 8, stride=8,
                                        padding=0, bias=False)),  # b, 64, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('2_e_batchnorm', nn.BatchNorm2d(64)),
                # -----------------------------------------------------------------------------
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
                ('4_e_conv2', nn.Conv2d(64, 16, 3, stride=1,
                                        padding=0, bias=False)),  # b, 16, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('5_e_batchnorm', nn.BatchNorm2d(16)),
                # -----------------------------------------------------------------------------
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(16, 64, 2,
                                                  stride=2, bias=False)),  # b, 64, 10, 10
                ('8_d_act1', nn.ReLU(True)),
                ('8_d_batchnorm', nn.BatchNorm2d(64)),
                # -----------------------------------------------------------------------------
                ('9_d_convt2', nn.ConvTranspose2d(64,
                                                  8,
                                                  3,
                                                  stride=1,
                                                  padding=0, bias=False)),  # b, 8, 12, 12
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(8, 1, 10, stride=2,
                                    padding=0, bias=False)),  # b, 1, 32, 32
                ('12_out', nn.Tanh())
                # -----------------------------------------------------------------------------
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_A7_Nopad_32_R_0_Small_filters_Inc(nn.Module):
    def __init__(self, dropout=0.0):
        super(autoencoder_A7_Nopad_32_R_0_Small_filters_Inc, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'A7_Nopad_32_R_0_Small_filters_Inc'
        self.dropout = nn.Dropout2d(dropout)
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 4, 5, stride=1,
                                        padding=0)),  # b, 64, 124, 124
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 62, 62
                ('4_e_conv2', nn.Conv2d(4, 8, 3, stride=1,
                                        padding=0)),  # b, 64, 60, 60
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=2)),  # b, 64, 30, 30
                ('7_e_conv3', nn.Conv2d(8, 16, 3, stride=1,
                                        padding=0)),  # b, 64, 28, 28
                ('8_e_act3', nn.ReLU(True)),
                ('9_e_maxpool3', nn.MaxPool2d(2, stride=2)),  # b, 64, 14, 14
                ('10_e_conv4', nn.Conv2d(16, 32, 3, stride=1,
                                         padding=0)),  # b, 64, 12, 12
                ('11_e_act4', nn.ReLU(True)),
                ('12_e_maxpool4', nn.MaxPool2d(2, stride=2)),  # b, 64, 6, 6
                ('13_e_conv5', nn.Conv2d(32, 64, 3, stride=1,
                                         padding=0)),  # b, 64, 4, 4
                ('14_e_act5', nn.ReLU(True))
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('16_d_convt1', nn.ConvTranspose2d(64, 16, 2,
                                                   stride=2,
                                                   padding=0)),  # b, 16, 8, 8
                ('17_d_act1', nn.ReLU(True)),
                ('18_d_convt2', nn.ConvTranspose2d(16, 4, 2,
                                                   stride=2,
                                                   padding=0)),  # b, 4, 16, 16
                ('19_d_act2', nn.ReLU(True)),
                ('20_d_convt3', nn.ConvTranspose2d(4, 1, 2,
                                                   stride=2,
                                                   padding=0)),  # b, 1, 32, 32
                ('21_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_A7_BN_Nopad_32_R_0_Small_filters_Inc(nn.Module):
    def __init__(self, dropout=0.0):
        super(autoencoder_A7_BN_Nopad_32_R_0_Small_filters_Inc, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'A7_BN_Nopad_32_R_0_Small_filters_Inc'
        self.dropout = nn.Dropout2d(dropout)
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 4, 5, stride=1,
                                        padding=0)),  # b, 4, 124, 124
                ('2_e_act1', nn.ReLU(True)),
                ('2_e_batchnorm', nn.BatchNorm2d(4)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 4, 62, 62
                ('4_e_conv2', nn.Conv2d(4, 8, 3, stride=1,
                                        padding=0)),  # b, 8, 60, 60
                ('5_e_act2', nn.ReLU(True)),
                ('5_e_batchnorm', nn.BatchNorm2d(8)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=2)),  # b, 8, 30, 30
                ('7_e_conv3', nn.Conv2d(8, 16, 3, stride=1,
                                        padding=0)),  # b, 16, 28, 28
                ('8_e_act3', nn.ReLU(True)),
                ('8_e_batchnorm', nn.BatchNorm2d(16)),
                ('9_e_maxpool3', nn.MaxPool2d(2, stride=2)),  # b, 16, 14, 14
                ('10_e_conv4', nn.Conv2d(16, 32, 3, stride=1,
                                         padding=0)),  # b, 32, 12, 12
                ('11_e_act4', nn.ReLU(True)),
                ('11_e_batchnorm', nn.BatchNorm2d(32)),
                ('12_e_maxpool4', nn.MaxPool2d(2, stride=2)),  # b, 32, 6, 6
                ('13_e_conv5', nn.Conv2d(32, 64, 3, stride=1,
                                         padding=0)),  # b, 64, 4, 4
                ('14_e_act5', nn.ReLU(True)),
                ('14_e_batchnorm', nn.BatchNorm2d(64)),
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('16_d_convt1', nn.ConvTranspose2d(64, 16, 2,
                                                   stride=2,
                                                   padding=0)),  # b, 16, 8, 8
                ('17_d_act1', nn.ReLU(True)),
                ('17_d_batchnorm', nn.BatchNorm2d(16)),
                ('18_d_convt2', nn.ConvTranspose2d(16, 4, 2,
                                                   stride=2,
                                                   padding=0)),  # b, 4, 16, 16
                ('19_d_act2', nn.ReLU(True)),
                ('19_d_batchnorm', nn.BatchNorm2d(4)),
                ('20_d_convt3', nn.ConvTranspose2d(4, 1, 2,
                                                   stride=2,
                                                   padding=0)),  # b, 1, 32, 32
                ('21_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_B1_Deep_32_0_Dense_Dec(nn.Module):
    def __init__(self, dropout=0.4):
        super(autoencoder_B1_Deep_32_0_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'B1_Deep_32_0_Dense_Dec'
        self.dropout = nn.Dropout2d(dropout)
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 64, 13, stride=13,
                                        padding=1)),  # b, 64, 10, 10
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 5, 5
                ('4_e_conv2', nn.Conv2d(64, 16, 3, stride=2,
                                        padding=1)),  # b, 16, 3, 3
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16, 2, 2
            ]))
        self.flat2mid = nn.Linear(16 * 2 * 2, 8)
        self.mid2unflat = nn.Linear(8, 16 * 2 * 2)
        self.decoder = nn.Sequential(
            OrderedDict([
                ('10_d_convt1', nn.ConvTranspose2d(16, 64, 2,
                                                   stride=2)),  # b, 64, 4, 4
                ('11_d_act1', nn.ReLU(True)),
                ('12_d_convt2',
                 nn.ConvTranspose2d(64, 8, 8, stride=2,
                                    padding=1)),  # b, 8, 12, 12
                ('13_d_act2', nn.ReLU(True)),
                ('14_d_convt3',
                 nn.ConvTranspose2d(8, 1, 12, stride=2,
                                    padding=1)),  # b, 1, 32, 32
                ('15_out', nn.Tanh())
            ]))
        self.flatten = Flatten()
        self.unflatten = UnFlatten()

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.dropout(x)
        y = self.flat2mid(x)
        x = self.mid2unflat(y)
        x = self.unflatten(x, sizec=16, sizex=2, sizey=2)
        x = self.decoder(x)
        return x, y


class autoencoder_B4_Deep_Nopad_32_0_Dense_Dec(nn.Module):
    def __init__(self, dropout=0.0):
        super(autoencoder_B4_Deep_Nopad_32_0_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'B4_Deep_Nopad_32_0_Dense_Dec'
        self.dropout = nn.Dropout2d(dropout)
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 64, 8, stride=8,
                                        padding=0)),  # b, 64, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
                ('4_e_conv2', nn.Conv2d(64, 16, 3, stride=1,
                                        padding=0)),  # b, 16, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.flat2mid = nn.Linear(16 * 5 * 5, 8)
        self.mid2unflat = nn.Linear(8, 16 * 5 * 5)

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(16, 64, 2,
                                                  stride=2)),  # b, 64, 10, 10
                ('8_d_act1', nn.ReLU(True)),
                ('9_d_convt2', nn.ConvTranspose2d(64,
                                                  8,
                                                  3,
                                                  stride=1,
                                                  padding=0)),  # b, 8, 12, 12
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(8, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('12_out', nn.Tanh())
            ]))
        self.flatten = Flatten()
        self.unflatten = UnFlatten()

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.dropout(x)
        y = self.flat2mid(x)
        x = self.mid2unflat(y)
        x = self.unflatten(x, sizec=16, sizex=5, sizey=5)
        x = self.decoder(x)
        return x, y


class autoencoder_C3_Deep_Nopad_32_Direct_out_R_0_Dense_Dec(nn.Module):
    def __init__(self, dropout=0.0):
        super(autoencoder_C3_Deep_Nopad_32_Direct_out_R_0_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'C3_Deep_Nopad_32_Direct_out_R_0_Dense_Dec'
        self.dropout = nn.Dropout2d(dropout)
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 64, 11, stride=9,
                                        padding=0)),  # b, 64, 14, 14
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 7, 7
                ('4_e_conv2', nn.Conv2d(64, 16, 3, stride=1,
                                        padding=0)),  # b, 16, 5, 5
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16, 4, 4
            ]))

        self.flat2mid = nn.Linear(16 * 4 * 4, 8)
        self.mid2unflat = nn.Linear(8, 32 * 32)
        self.flatten = Flatten()
        self.unflatten = UnFlatten()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.dropout(x)
        y = self.flat2mid(x)
        x = self.mid2unflat(y)
        x = self.unflatten(x, sizec=1, sizex=32, sizey=32)
        x = self.tanh(x)
        return x, y

# OLD NAME Nopad_32_Direct_out_R_1_Dense_Dec


class autoencoder_C2_Nopad_32_Direct_out_R_0_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_C2_Nopad_32_Direct_out_R_0_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'C2_Nopad_32_Direct_out_R_0_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 64, 3, stride=1,
                                        padding=0)),  # b, 64, 126, 126
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2,
                                              padding=1)),  # b, 64, 64, 64
                ('4_e_conv2', nn.Conv2d(64, 16, 3, stride=1,
                                        padding=1)),  # b, 16, 64, 64
                ('5_e_act2', nn.ReLU(True)),
                # ('6_e_maxpool2', nn.MaxPool2d(2, stride=2,
                #                               padding=1))  # b, 16, 33, 33
                ('4_e_conv2', nn.Conv2d(64, 1, 2, stride=1,
                                        padding=0)),  # b, 16, 32, 32
                ('5_e_act2', nn.Tanh()),
            ]))

    def forward(self, x):
        x = self.encoder(x)
        return x, 1


class autoencoder_D1_Deep_Nopad_32_R_0_Small_filters_Inc(nn.Module):
    def __init__(self, dropout=0.0):
        super(autoencoder_D1_Deep_Nopad_32_R_0_Small_filters_Inc, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'D1_Deep_Nopad_32_R_0_Small_filters_Inc'
        self.dropout = nn.Dropout2d(dropout)
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 4, 5, stride=1,
                                        padding=0)),  # b, 64, 124, 124
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 62, 62
                ('4_e_conv2', nn.Conv2d(4, 8, 3, stride=1,
                                        padding=0)),  # b, 64, 60, 60
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=2)),  # b, 64, 30, 30
                ('7_e_conv3', nn.Conv2d(8, 16, 3, stride=1,
                                        padding=0)),  # b, 64, 28, 28
                ('8_e_act3', nn.ReLU(True)),
                ('9_e_maxpool3', nn.MaxPool2d(2, stride=2)),  # b, 64, 14, 14
                ('10_e_conv4', nn.Conv2d(16, 32, 3, stride=1,
                                         padding=0)),  # b, 64, 12, 12
                ('11_e_act4', nn.ReLU(True)),
                ('12_e_maxpool4', nn.MaxPool2d(2, stride=2)),  # b, 64, 6, 6
                ('13_e_conv5', nn.Conv2d(32, 64, 3, stride=1,
                                         padding=0)),  # b, 64, 4, 4
                ('14_e_act5', nn.ReLU(True)),
                ('14_e_maxpool5', nn.MaxPool2d(2, stride=2)),  # b, 64, 14, 14
            ]))

        self.flat_to_l1 = nn.Linear(64 * 2 * 2, 32)
        self.l1_to_l2 = nn.Linear(32, 12)
        self.l2_to_l3 = nn.Linear(12, 32)
        self.l3_to_l4 = nn.Linear(32, 4 * 4 * 4)
        self.unflatten = UnFlatten()

        self.decoder = nn.Sequential(
            OrderedDict([
                ('16_d_convt1', nn.ConvTranspose2d(4, 3, 2,
                                                   stride=2,
                                                   padding=0)),  # b, 4, 8, 8
                ('17_d_act1', nn.ReLU(True)),
                ('18_d_convt2', nn.ConvTranspose2d(3, 2, 2,
                                                   stride=2,
                                                   padding=0)),  # b, 3, 16, 16
                ('19_d_act2', nn.ReLU(True)),
                ('20_d_convt3', nn.ConvTranspose2d(2, 1, 2,
                                                   stride=2,
                                                   padding=0)),  # b, 1, 2, 32
                ('21_out', nn.Tanh())
            ]))
        self.flatten = Flatten()
        self.unflatten = UnFlatten()

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = F.relu(self.flat_to_l1(x))
        y = F.relu(self.l1_to_l2(x))
        x = self.dropout(y)
        x = F.relu(self.l2_to_l3(x))
        x = F.relu(self.l3_to_l4(x))
        x = self.unflatten(x, sizec=4, sizex=4, sizey=4)
        x = self.decoder(x)
        return x, y


class autoencoder_D1_Deep_Nopad_32_R_0_Small_filters_Inc_no_bias(nn.Module):
    def __init__(self, dropout=0.0):
        super(autoencoder_D1_Deep_Nopad_32_R_0_Small_filters_Inc_no_bias,
              self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'D1_Deep_Nopad_32_R_0_Small_filters_Inc_no_bias'
        self.dropout = nn.Dropout2d(dropout)
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 4, 5, stride=1,
                                        padding=0, bias=False)),  # b, 64, 124, 124
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 62, 62
                ('4_e_conv2', nn.Conv2d(4, 8, 3, stride=1,
                                        padding=0, bias=False)),  # b, 64, 60, 60
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=2)),  # b, 64, 30, 30
                ('7_e_conv3', nn.Conv2d(8, 16, 3, stride=1,
                                        padding=0, bias=False)),  # b, 64, 28, 28
                ('8_e_act3', nn.ReLU(True)),
                ('9_e_maxpool3', nn.MaxPool2d(2, stride=2)),  # b, 64, 14, 14
                ('10_e_conv4', nn.Conv2d(16, 32, 3, stride=1,
                                         padding=0, bias=False)),  # b, 64, 12, 12
                ('11_e_act4', nn.ReLU(True)),
                ('12_e_maxpool4', nn.MaxPool2d(2, stride=2)),  # b, 64, 6, 6
                ('13_e_conv5', nn.Conv2d(32, 64, 3, stride=1,
                                         padding=0, bias=False)),  # b, 64, 4, 4
                ('14_e_act5', nn.ReLU(True)),
                ('14_e_maxpool5', nn.MaxPool2d(2, stride=2)),  # b, 64, 14, 14
            ]))

        self.flat_to_l1 = nn.Linear(64 * 2 * 2, 32)
        self.l1_to_l2 = nn.Linear(32, 12)
        self.l2_to_l3 = nn.Linear(12, 32)
        self.l3_to_l4 = nn.Linear(32, 4 * 4 * 4)
        self.unflatten = UnFlatten()

        self.decoder = nn.Sequential(
            OrderedDict([
                ('16_d_convt1', nn.ConvTranspose2d(4, 3, 2,
                                                   stride=2,
                                                   padding=0, bias=False)),  # b, 4, 8, 8
                ('17_d_act1', nn.ReLU(True)),
                ('18_d_convt2', nn.ConvTranspose2d(3, 2, 2,
                                                   stride=2,
                                                   padding=0, bias=False)),  # b, 3, 16, 16
                ('19_d_act2', nn.ReLU(True)),
                ('20_d_convt3', nn.ConvTranspose2d(2, 1, 2,
                                                   stride=2,
                                                   padding=0, bias=False)),  # b, 1, 2, 32
                ('21_out', nn.Tanh())
            ]))
        self.flatten = Flatten()
        self.unflatten = UnFlatten()

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = F.relu(self.flat_to_l1(x))
        y = F.relu(self.l1_to_l2(x))
        x = self.dropout(y)
        x = F.relu(self.l2_to_l3(x))
        x = F.relu(self.l3_to_l4(x))
        x = self.unflatten(x, sizec=4, sizex=4, sizey=4)
        x = self.decoder(x)
        return x, y


class autoencoder_D1_Deep_Nopad_32_R_0_Small_filters_Inc_no_bias_BN(nn.Module):
    def __init__(self, dropout=0.0):
        super(
            autoencoder_D1_Deep_Nopad_32_R_0_Small_filters_Inc_no_bias_BN, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'D1_Deep_Nopad_32_R_0_Small_filters_Inc_no_bias_BN'
        self.dropout = nn.Dropout2d(dropout)
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 4, 5, stride=1,
                                        padding=0, bias=False)),  # b, 64, 124, 124
                ('2_e_act1', nn.ReLU(True)),
                ('2_e_batchnorm', nn.BatchNorm2d(4)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 62, 62
                ('4_e_conv2', nn.Conv2d(4, 8, 3, stride=1,
                                        padding=0, bias=False)),  # b, 64, 60, 60
                ('5_e_act2', nn.ReLU(True)),
                ('5_e_batchnorm', nn.BatchNorm2d(8)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=2)),  # b, 64, 30, 30
                ('7_e_conv3', nn.Conv2d(8, 16, 3, stride=1,
                                        padding=0, bias=False)),  # b, 64, 28, 28
                ('8_e_act3', nn.ReLU(True)),
                ('8_e_batchnorm', nn.BatchNorm2d(16)),
                ('9_e_maxpool3', nn.MaxPool2d(2, stride=2)),  # b, 64, 14, 14
                ('10_e_conv4', nn.Conv2d(16, 32, 3, stride=1,
                                         padding=0, bias=False)),  # b, 64, 12, 12
                ('11_e_act4', nn.ReLU(True)),
                ('11_e_batchnorm', nn.BatchNorm2d(32)),
                ('12_e_maxpool4', nn.MaxPool2d(2, stride=2)),  # b, 64, 6, 6
                ('13_e_conv5', nn.Conv2d(32, 64, 3, stride=1,
                                         padding=0, bias=False)),  # b, 64, 4, 4
                ('14_e_act5', nn.ReLU(True)),
                ('14_e_batchnorm', nn.BatchNorm2d(64)),
                ('14_e_maxpool5', nn.MaxPool2d(2, stride=2)),  # b, 64, 14, 14
            ]))

        self.flat_to_l1 = nn.Linear(64 * 2 * 2, 32)
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.l1_to_l2 = nn.Linear(32, 12)
        self.bn2 = nn.BatchNorm1d(num_features=12)
        self.l2_to_l3 = nn.Linear(12, 32)
        self.bn3 = nn.BatchNorm1d(num_features=32)
        self.l3_to_l4 = nn.Linear(32, 4 * 4 * 4)
        self.bn4 = nn.BatchNorm1d(num_features=64)
        self.unflatten = UnFlatten()

        self.decoder = nn.Sequential(
            OrderedDict([
                ('16_d_convt1', nn.ConvTranspose2d(4, 3, 2,
                                                   stride=2,
                                                   padding=0, bias=False)),  # b, 4, 8, 8
                ('17_d_act1', nn.ReLU(True)),
                ('17_d_batchnorm', nn.BatchNorm2d(3)),
                ('18_d_convt2', nn.ConvTranspose2d(3, 2, 2,
                                                   stride=2,
                                                   padding=0, bias=False)),  # b, 3, 16, 16
                ('19_d_act2', nn.ReLU(True)),
                ('19_d_batchnorm', nn.BatchNorm2d(2)),
                ('20_d_convt3', nn.ConvTranspose2d(2, 1, 2,
                                                   stride=2,
                                                   padding=0, bias=False)),  # b, 1, 2, 32
                ('21_out', nn.Tanh())
            ]))
        self.flatten = Flatten()
        self.unflatten = UnFlatten()

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.bn1(F.relu(self.flat_to_l1(x)))
        y = self.bn2(F.relu(self.l1_to_l2(x)))
        x = self.dropout(y)
        x = self.bn3(F.relu(self.l2_to_l3(x)))
        x = self.bn4(F.relu(self.l3_to_l4(x)))
        x = self.unflatten(x, sizec=4, sizex=4, sizey=4)
        x = self.decoder(x)
        return x, y


class autoencoder_A4_R_Nopad_64_0_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_A4_R_Nopad_64_0_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'A4_R_Nopad_64_0_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 64, 8, stride=8,
                                        padding=0)),  # b, 64, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
                ('4_e_conv2', nn.Conv2d(64, 16, 3, stride=1,
                                        padding=0)),  # b, 16, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(16, 64, 2,
                                                  stride=2)),  # b, 64, 10, 10
                ('8_d_act1', nn.ReLU(True)),
                ('9_d_convt2', nn.ConvTranspose2d(64,
                                                  8,
                                                  3,
                                                  stride=1,
                                                  padding=0)),  # b, 8, 12, 12
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(8, 4, 6, stride=2,
                                    padding=0)),  # b, 1, 28, 28
                ('12_d_act3', nn.ReLU(True)),
                ('13_d_convt4',
                 nn.ConvTranspose2d(4, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 64, 64
                ('14_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_A4_R_Nopad_128_0_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_A4_R_Nopad_128_0_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'A4_R_Nopad_64_0_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 64, 8, stride=8,
                                        padding=0)),  # b, 64, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
                ('4_e_conv2', nn.Conv2d(64, 16, 3, stride=1,
                                        padding=0)),  # b, 16, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(16, 64, 2,
                                                  stride=2)),  # b, 64, 10, 10
                ('8_d_act1', nn.ReLU(True)),
                ('9_d_convt2', nn.ConvTranspose2d(64,
                                                  8,
                                                  3,
                                                  stride=1,
                                                  padding=0)),  # b, 8, 12, 12
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3', nn.ConvTranspose2d(8, 4, 5, stride=2,
                                                   padding=0)),  # b, 4, 27, 27
                ('12_d_act3', nn.ReLU(True)),
                ('13_d_convt4',
                 nn.ConvTranspose2d(4, 2, 8, stride=2,
                                    padding=0)),  # b, 2, 60, 60
                ('14_d_act5', nn.ReLU(True)),
                ('15_d_convt5',
                 nn.ConvTranspose2d(2, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 128, 128
                ('16_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_NA1_R_Nopad_32_0_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_NA1_R_Nopad_32_0_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'NA1_R_Nopad_32_0_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(3, 256, 5, stride=3,
                                        padding=0)),  # b, 256, 42, 42
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_conv2', nn.Conv2d(256, 128, 2, stride=2,
                                        padding=0)),  # b, 128, 21, 21
                # ('4_e_act4', nn.ReLU(True)),
                ('5_e_conv3', nn.Conv2d(128, 64, 3, stride=2,
                                        padding=0)),  # b, 64, 10, 10
                ('6_e_act3', nn.ReLU(True)),
                ('3_e_Avgpool1', nn.AvgPool2d(2, stride=1))  # b, 64, 9, 9
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(64, 32, 2,
                                                  stride=1)),  # b, 32, 10, 10
                ('8_d_act1', nn.ReLU(True)),
                ('9_d_convt2', nn.ConvTranspose2d(32,
                                                  16,
                                                  5,
                                                  stride=1,
                                                  padding=0)),  # b, 32, 14, 14
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(16, 1, 6, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('12_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_NA1_R_Nopad_32_1_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_NA1_R_Nopad_32_1_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'NA1_R_Nopad_32_1_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(3, 64, 5, stride=3,
                                        padding=0)),  # b, 256, 42, 42
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_conv2', nn.Conv2d(64, 32, 2, stride=2,
                                        padding=0)),  # b, 128, 21, 21
                # ('4_e_act4', nn.ReLU(True)),
                ('5_e_conv3', nn.Conv2d(32, 16, 3, stride=2,
                                        padding=0)),  # b, 64, 10, 10
                ('6_e_act3', nn.ReLU(True)),
                ('3_e_Avgpool1', nn.AvgPool2d(2, stride=1))  # b, 64, 9, 9
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(16, 8, 2,
                                                  stride=1)),  # b, 32, 10, 10
                ('8_d_act1', nn.ReLU(True)),
                ('9_d_convt2', nn.ConvTranspose2d(8,
                                                  4,
                                                  5,
                                                  stride=1,
                                                  padding=0)),  # b, 32, 14, 14
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(4, 1, 6, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('12_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_NA1_R_Nopad_32_2_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_NA1_R_Nopad_32_2_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'NA1_R_Nopad_32_2_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(3, 16, 5, stride=3,
                                        padding=0)),  # b, 256, 42, 42
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_conv2', nn.Conv2d(16, 32, 2, stride=2,
                                        padding=0)),  # b, 128, 21, 21
                # ('4_e_act4', nn.ReLU(True)),
                ('5_e_conv3', nn.Conv2d(32, 64, 3, stride=2,
                                        padding=0)),  # b, 64, 10, 10
                ('6_e_act3', nn.ReLU(True)),
                ('3_e_Avgpool1', nn.AvgPool2d(2, stride=1))  # b, 64, 9, 9
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(64, 32, 2,
                                                  stride=1)),  # b, 32, 10, 10
                ('8_d_act1', nn.ReLU(True)),
                ('9_d_convt2', nn.ConvTranspose2d(32,
                                                  16,
                                                  5,
                                                  stride=1,
                                                  padding=0)),  # b, 32, 14, 14
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(16, 1, 6, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('12_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_NA3_R_Nopad_32_0_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_NA3_R_Nopad_32_0_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'NA3_R_Nopad_32_0_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('0_e_conv1', nn.Conv2d(3, 16, 1, stride=1,
                                        padding=0)),  # b, 16, 128, 128
                ('1_e_act1', nn.ReLU(True)),
                ('2_e_conv2', nn.Conv2d(16, 256, 5, stride=3,
                                        padding=0)),  # b, 256, 42, 42
                ('3_e_act2', nn.ReLU(True)),
                ('4_e_conv3', nn.Conv2d(256, 128, 2, stride=2,
                                        padding=0)),  # b, 128, 21, 21
                # ('4_e_act4', nn.ReLU(True)),
                ('5_e_conv4', nn.Conv2d(128, 64, 3, stride=2,
                                        padding=0)),  # b, 64, 10, 10
                ('6_e_act4', nn.ReLU(True)),
                ('7_e_Avgpool1', nn.AvgPool2d(2, stride=1))  # b, 64, 9, 9
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('8_d_convt1', nn.ConvTranspose2d(64, 32, 2,
                                                  stride=1)),  # b, 32, 10, 10
                ('9_d_act1', nn.ReLU(True)),
                ('10_d_convt2', nn.ConvTranspose2d(32,
                                                   16,
                                                   5,
                                                   stride=1,
                                                   padding=0)),  # b, 32, 14, 14
                ('11_d_act2', nn.ReLU(True)),
                ('12_d_convt3',
                 nn.ConvTranspose2d(16, 1, 6, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('13_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_NA1_R_Nopad_32_2_Dense_Dec_upscale2(nn.Module):
    def __init__(self):
        super(autoencoder_NA1_R_Nopad_32_2_Dense_Dec_upscale2, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'NA1_R_Nopad_32_2_Dense_Dec_upscale2'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(3, 16, 5, stride=3,
                                        padding=0)),  # b, 256, 42, 42
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_conv2', nn.Conv2d(16, 32, 2, stride=2,
                                        padding=0)),  # b, 128, 21, 21
                # ('4_e_act4', nn.ReLU(True)),
                ('5_e_conv3', nn.Conv2d(32, 64, 3, stride=2,
                                        padding=0)),  # b, 64, 10, 10
                ('6_e_act3', nn.ReLU(True)),
                ('3_e_Avgpool1', nn.AvgPool2d(2, stride=1))  # b, 64, 9, 9
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(64, 32, 2,
                                                  stride=1)),  # b, 32, 10, 10
                ('8_d_act1', nn.ReLU(True)),
                ('9_d_convt2', nn.ConvTranspose2d(32,
                                                  16,
                                                  5,
                                                  stride=1,
                                                  padding=0)),  # b, 32, 14, 14
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(16, 1, 6, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('12_d_upscale',
                 Upsample(scale_factor=4, mode='bilinear', align_corners=True)),
                ('13_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_unettype1(nn.Module):
    def __init__(self):
        super(autoencoder_unettype1, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'autoencoder_unettype1'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 16, 5, stride=3,
                                        padding=0)),
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_conv1', nn.Conv2d(16, 32, 5, stride=1,
                                        padding=0)),
                ('4_e_act1', nn.ReLU(True)),
                ('5_e_maxpool1', nn.MaxPool2d(3, stride=1, padding=1)),

                ('6_e_conv1', nn.Conv2d(32, 64, 5, stride=3,
                                        padding=0)),
                ('7_e_act1', nn.ReLU(True)),
                ('8_e_conv1', nn.Conv2d(64, 128, 5, stride=1,
                                        padding=0)),
                ('9_e_act1', nn.ReLU(True)),
                ('10_e_maxpool1', nn.MaxPool2d(
                    3, stride=1, padding=1)),
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('11_d_upscale',
                 Upsample(scale_factor=2, mode='bilinear', align_corners=True)),
                ('12_e_conv1', nn.Conv2d(128, 64, 3, stride=1,
                                         padding=1)),
                ('13_e_act1', nn.ReLU(True)),
                ('14_e_conv1', nn.Conv2d(64, 32, 3, stride=1,
                                         padding=1)),
                ('15_e_act1', nn.ReLU(True)),

                ('16_d_upscale',
                 Upsample(scale_factor=2, mode='bilinear', align_corners=True)),
                ('17_e_conv1', nn.Conv2d(32, 16, 3, stride=1,
                                         padding=1)),
                ('18_e_act1', nn.ReLU(True)),
                ('19_e_conv1', nn.Conv2d(16, 8, 3, stride=1,
                                         padding=1)),
                ('20_e_act1', nn.ReLU(True)),

                ('21_d_upscale',
                 Upsample(scale_factor=2, mode='bilinear', align_corners=True)),
                ('22_e_conv1', nn.Conv2d(8, 4, 3, stride=1,
                                         padding=1)),
                ('23_e_act1', nn.ReLU(True)),
                ('24_e_conv1', nn.Conv2d(4, 2, 3, stride=1,
                                         padding=1)),
                ('25_e_act1', nn.ReLU(True)),

                ('26_d_upscale',
                 Upsample(scale_factor=2, mode='bilinear', align_corners=True)),
                ('27_e_conv1', nn.Conv2d(2, 1, 3, stride=1,
                                         padding=1)),
                ('28_e_act1', nn.ReLU(True)),
                ('29_e_conv1', nn.Conv2d(1, 1, 3, stride=1,
                                         padding=1)),
                ('30_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_unettype1_32(nn.Module):
    def __init__(self):
        super(autoencoder_unettype1_32, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'autoencoder_unettype1_32'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 16, 5, stride=3,
                                        padding=0)),
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_conv1', nn.Conv2d(16, 16, 5, stride=1,
                                        padding=0)),
                ('4_e_act1', nn.ReLU(True)),
                ('5_e_maxpool1', nn.MaxPool2d(3, stride=1, padding=1)),

                ('6_e_conv1', nn.Conv2d(16, 32, 5, stride=3,
                                        padding=0)),
                ('7_e_act1', nn.ReLU(True)),
                ('8_e_conv1', nn.Conv2d(32, 32, 5, stride=1,
                                        padding=0)),
                ('9_e_act1', nn.ReLU(True)),
                ('10_e_maxpool1', nn.MaxPool2d(
                    3, stride=1, padding=1)),
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('11_d_upscale',
                 Upsample(scale_factor=2, mode='bilinear', align_corners=True)),
                ('12_e_conv1', nn.Conv2d(32, 16, 3, stride=1,
                                         padding=1)),
                ('13_e_act1', nn.ReLU(True)),
                ('14_d_upscale',
                 Upsample(scale_factor=2, mode='bilinear', align_corners=True)),
                ('15_e_conv1', nn.Conv2d(16, 1, 3, stride=1,
                                         padding=1)),
                ('16_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_A4_R_Nopad_32_u0_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_A4_R_Nopad_32_u0_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'autoencoder_A4_R_Nopad_32_u0_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 64, 8, stride=8,
                                        padding=0)),  # b, 64, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
                ('4_e_conv2', nn.Conv2d(64, 16, 3, stride=1,
                                        padding=0)),  # b, 16, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_upscale',
                 Upsample(scale_factor=2, mode='bilinear', align_corners=True)),
                # ('8_d_act2', nn.ReLU(True)),
                ('9_d_convt2', nn.ConvTranspose2d(16,
                                                  8,
                                                  3,
                                                  stride=1,
                                                  padding=0)),  # b, 8, 12, 12
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(8, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('12_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_TA7_Nopad_32_R_0_Small_filters_Inc(nn.Module):
    def __init__(self, dropout=0.0):
        super(autoencoder_TA7_Nopad_32_R_0_Small_filters_Inc, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'TA7_Nopad_32_R_0_Small_filters_Inc'
        self.dropout = nn.Dropout2d(dropout)
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 4, 5, stride=1,
                                        padding=0)),  # b, 64, 124, 124
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 62, 62
                ('4_e_conv2', nn.Conv2d(4, 8, 3, stride=1,
                                        padding=0)),  # b, 64, 60, 60
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=2)),  # b, 64, 30, 30
                ('7_e_conv3', nn.Conv2d(8, 16, 3, stride=1,
                                        padding=0)),  # b, 64, 28, 28
                ('8_e_act3', nn.ReLU(True)),
                ('9_e_maxpool3', nn.MaxPool2d(2, stride=2)),  # b, 64, 14, 14
                ('10_e_conv4', nn.Conv2d(16, 32, 3, stride=1,
                                         padding=0)),  # b, 64, 12, 12
                ('11_e_act4', nn.ReLU(True)),
                ('12_e_maxpool4', nn.MaxPool2d(2, stride=2)),  # b, 64, 6, 6
                ('13_e_conv5', nn.Conv2d(32, 64, 2, stride=1,
                                         padding=0)),  # b, 64, 4, 4
                ('14_e_act5', nn.ReLU(True))
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d_convt1', nn.ConvTranspose2d(64, 8, 2,
                                                  stride=2)),  # b, 64, 10, 10
                ('8_d_act1', nn.ReLU(True)),
                ('9_d_convt2', nn.ConvTranspose2d(8,
                                                  4,
                                                  3,
                                                  stride=1,
                                                  padding=0)),  # b, 8, 12, 12
                ('10_d_act2', nn.ReLU(True)),
                ('11_d_convt3',
                 nn.ConvTranspose2d(4, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('12_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_TA8_Nopad_32_R_0_Small_filters_Inc(nn.Module):
    def __init__(self, dropout=0.0):
        super(autoencoder_TA8_Nopad_32_R_0_Small_filters_Inc, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'TA8_Nopad_32_R_0_Small_filters_Inc'
        self.dropout = nn.Dropout2d(dropout)
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 4, 5, stride=3,
                                        padding=0)),  # b, 64, 124, 124
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_conv2', nn.Conv2d(4, 8, 3, stride=1,
                                        padding=0)),  # b, 64, 60, 60
                ('4_e_act2', nn.ReLU(True)),
                ('5_e_conv3', nn.Conv2d(8, 16, 3, stride=1,
                                        padding=0)),  # b, 64, 28, 28
                ('6_e_act3', nn.ReLU(True)),
                ('7_e_maxpool4', nn.MaxPool2d(2, stride=2)),  # b, 64, 6, 6
                ('8_e_conv4', nn.Conv2d(16, 32, 3, stride=2,
                                        padding=0)),  # b, 64, 12, 12
                ('9_e_act4', nn.ReLU(True)),
                ('10_e_conv4', nn.Conv2d(32, 64, 3, stride=1,
                                         padding=0)),  # b, 64, 12, 12
                ('11_e_act4', nn.ReLU(True)),
                ('12_e_conv5', nn.Conv2d(64, 64, 3, stride=1,
                                         padding=0)),  # b, 64, 4, 4
                ('13_e_act5', nn.ReLU(True))
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('14_d_convt1', nn.ConvTranspose2d(64, 8, 2,
                                                   stride=2)),  # b, 64, 10, 10
                ('15_d_act1', nn.ReLU(True)),
                ('16_d_convt2', nn.ConvTranspose2d(8,
                                                   4,
                                                   3,
                                                   stride=1,
                                                   padding=0)),  # b, 8, 12, 12
                ('17_d_act2', nn.ReLU(True)),
                ('18_d_convt3',
                 nn.ConvTranspose2d(4, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('19_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_TA3_Nopad_32_R_0_Small_filters_Inc(nn.Module):
    def __init__(self, dropout=0.0):
        super(autoencoder_TA3_Nopad_32_R_0_Small_filters_Inc, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'TA3_Nopad_32_R_0_Small_filters_Inc'
        self.dropout = nn.Dropout2d(dropout)
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 16, 5, stride=3,
                                        padding=0)),  # b, 16, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 16, 8, 8
                ('4_e_conv2', nn.Conv2d(16, 64, 3, stride=2,
                                        padding=0)),  # b, 64, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=2))  # b, 64, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('14_d_convt1', nn.ConvTranspose2d(64, 8, 2,
                                                   stride=2)),  # b, 64, 10, 10
                ('15_d_act1', nn.ReLU(True)),
                ('16_d_convt2', nn.ConvTranspose2d(8,
                                                   4,
                                                   3,
                                                   stride=1,
                                                   padding=0)),  # b, 8, 12, 12
                ('17_d_act2', nn.ReLU(True)),
                ('18_d_convt3',
                 nn.ConvTranspose2d(4, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('19_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_TA3_Nopad_32_R_1_Small_filters_Inc(nn.Module):
    def __init__(self, dropout=0.0):
        super(autoencoder_TA3_Nopad_32_R_1_Small_filters_Inc, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'TA3_Nopad_32_R_1_Small_filters_Inc'
        self.dropout = nn.Dropout2d(dropout)
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 20, 5, stride=3,
                                        padding=0)),  # b, 16, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 16, 8, 8
                ('4_e_conv2', nn.Conv2d(20, 64, 3, stride=2,
                                        padding=0)),  # b, 64, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=2))  # b, 64, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('14_d_convt1', nn.ConvTranspose2d(64, 8, 2,
                                                   stride=2)),  # b, 64, 10, 10
                ('15_d_act1', nn.ReLU(True)),
                ('16_d_convt2', nn.ConvTranspose2d(8,
                                                   4,
                                                   3,
                                                   stride=1,
                                                   padding=0)),  # b, 8, 12, 12
                ('17_d_act2', nn.ReLU(True)),
                ('18_d_convt3',
                 nn.ConvTranspose2d(4, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('19_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_TA2_Nopad_32_R_0_Small_filters_Inc(nn.Module):
    def __init__(self, dropout=0.0):
        super(autoencoder_TA2_Nopad_32_R_0_Small_filters_Inc, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'TA2_Nopad_32_R_0_Small_filters_Inc'
        self.dropout = nn.Dropout2d(dropout)
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 16, 8, stride=6,
                                        padding=0)),  # b, 16, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(3, stride=2)),  # b, 16, 8, 8
                ('4_e_conv2', nn.Conv2d(16, 64, 4, stride=1,
                                        padding=0)),  # b, 64, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(3, stride=1))  # b, 64, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('14_d_convt1', nn.ConvTranspose2d(64, 8, 2,
                                                   stride=2)),  # b, 64, 10, 10
                ('15_d_act1', nn.ReLU(True)),
                ('16_d_convt2', nn.ConvTranspose2d(8,
                                                   4,
                                                   3,
                                                   stride=1,
                                                   padding=0)),  # b, 8, 12, 12
                ('17_d_act2', nn.ReLU(True)),
                ('18_d_convt3',
                 nn.ConvTranspose2d(4, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('19_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_TA2_Nopad_32_R_1_Small_filters_Inc(nn.Module):
    def __init__(self, dropout=0.0):
        super(autoencoder_TA2_Nopad_32_R_1_Small_filters_Inc, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'TA2_Nopad_32_R_1_Small_filters_Inc'
        self.dropout = nn.Dropout2d(dropout)
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 16, 8, stride=6,
                                        padding=0)),  # b, 16, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_conv2', nn.Conv2d(16, 16, 3, stride=2,
                                        padding=0)),  # b, 16, 8, 8
                ('4_e_act2', nn.ReLU(True)),
                ('5_e_conv3', nn.Conv2d(16, 64, 4, stride=1,
                                        padding=0)),  # b, 64, 6, 6
                ('6_e_act3', nn.ReLU(True)),
                ('7_e_maxpool2', nn.MaxPool2d(3, stride=1))  # b, 64, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('8_d_convt1', nn.ConvTranspose2d(64, 8, 2,
                                                  stride=2)),  # b, 64, 10, 10
                ('9_d_act1', nn.ReLU(True)),
                ('10_d_convt2', nn.ConvTranspose2d(8,
                                                   4,
                                                   3,
                                                   stride=1,
                                                   padding=0)),  # b, 8, 12, 12
                ('11_d_act2', nn.ReLU(True)),
                ('12_d_convt3',
                 nn.ConvTranspose2d(4, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('19_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_TA2_Nopad_32_R_2_Small_filters_Inc(nn.Module):
    def __init__(self, dropout=0.0):
        super(autoencoder_TA2_Nopad_32_R_2_Small_filters_Inc, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'TA2_Nopad_32_R_2_Small_filters_Inc'
        self.dropout = nn.Dropout2d(dropout)
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 16, 8, stride=3,
                                        padding=0)),  # b, 16, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(3, stride=2)),  # b, 16, 8, 8
                ('4_e_conv2', nn.Conv2d(16, 64, 2, stride=2,
                                        padding=0)),  # b, 64, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=2))  # b, 64, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('14_d_convt1', nn.ConvTranspose2d(64, 8, 2,
                                                   stride=2)),  # b, 64, 10, 10
                ('15_d_act1', nn.ReLU(True)),
                ('16_d_convt2', nn.ConvTranspose2d(8,
                                                   4,
                                                   3,
                                                   stride=1,
                                                   padding=0)),  # b, 8, 12, 12
                ('17_d_act2', nn.ReLU(True)),
                ('18_d_convt3',
                 nn.ConvTranspose2d(4, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('19_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_TA2_Nopad_32_R_3_Small_filters_Inc(nn.Module):
    def __init__(self, dropout=0.0):
        super(autoencoder_TA2_Nopad_32_R_3_Small_filters_Inc, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'TA2_Nopad_32_R_3_Small_filters_Inc'
        self.dropout = nn.Dropout2d(dropout)
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 16, 8, stride=3,
                                        padding=0)),  # b, 16, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_conv2', nn.Conv2d(16, 16, 3, stride=2,
                                        padding=0)),  # b, 16, 8, 8
                ('4_e_act2', nn.ReLU(True)),
                ('5_e_conv3', nn.Conv2d(16, 64, 2, stride=2,
                                        padding=0)),  # b, 64, 6, 6
                ('6_e_act3', nn.ReLU(True)),
                ('7_e_maxpool2', nn.MaxPool2d(2, stride=2))  # b, 64, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('8_d_convt1', nn.ConvTranspose2d(64, 8, 2,
                                                  stride=2)),  # b, 64, 10, 10
                ('9_d_act1', nn.ReLU(True)),
                ('10_d_convt2', nn.ConvTranspose2d(8,
                                                   4,
                                                   3,
                                                   stride=1,
                                                   padding=0)),  # b, 8, 12, 12
                ('11_d_act2', nn.ReLU(True)),
                ('12_d_convt3',
                 nn.ConvTranspose2d(4, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('19_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_TA9_R_Nopad_32_0_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_TA9_R_Nopad_32_0_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'TA9_R_Nopad_32_0_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 16, 8, stride=8,
                                        padding=0)),  # b, 64, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_conv1', nn.Conv2d(16, 16, 2, stride=2,
                                        padding=0)),  # b, 64, 16, 16
                ('4_e_act1', nn.ReLU(True)),
                ('5_e_conv2', nn.Conv2d(16, 64, 3, stride=1,
                                        padding=0)),  # b, 16, 6, 6
                ('6_e_act2', nn.ReLU(True)),
                ('7_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('8_d_convt1', nn.ConvTranspose2d(64, 8, 2,
                                                  stride=2)),  # b, 64, 10, 10
                ('9_d_act1', nn.ReLU(True)),
                ('10_d_convt2', nn.ConvTranspose2d(8,
                                                   4,
                                                   3,
                                                   stride=1,
                                                   padding=0)),  # b, 8, 12, 12
                ('11_d_act2', nn.ReLU(True)),
                ('12_d_convt3',
                 nn.ConvTranspose2d(4, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('13_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_TA8_Nopad_32_R_1_Small_filters_Inc(nn.Module):
    def __init__(self, dropout=0.0):
        super(autoencoder_TA8_Nopad_32_R_1_Small_filters_Inc, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'TA8_Nopad_32_R_1_Small_filters_Inc'
        self.dropout = nn.Dropout2d(dropout)
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 4, 64, stride=1,
                                        padding=0)),  # b, 64, 124, 124
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_conv2', nn.Conv2d(4, 8, 32, stride=1,
                                        padding=0)),  # b, 64, 60, 60
                ('4_e_act2', nn.ReLU(True)),
                ('5_e_conv3', nn.Conv2d(8, 16, 16, stride=1,
                                        padding=0)),  # b, 64, 28, 28
                ('6_e_act3', nn.ReLU(True)),
                ('7_e_conv4', nn.Conv2d(16, 32, 10, stride=1,
                                        padding=0)),  # b, 64, 12, 12
                ('8_e_act4', nn.ReLU(True)),
                ('9_e_conv4', nn.Conv2d(32, 64, 5, stride=1,
                                        padding=0)),  # b, 64, 12, 12
                ('10_e_act4', nn.ReLU(True)),
                ('11_e_maxpool4', nn.MaxPool2d(2, stride=1)),  # b, 64, 6, 6
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('14_d_convt1', nn.ConvTranspose2d(64, 8, 2,
                                                   stride=2)),  # b, 64, 10, 10
                ('15_d_act1', nn.ReLU(True)),
                ('16_d_convt2', nn.ConvTranspose2d(8,
                                                   4,
                                                   3,
                                                   stride=1,
                                                   padding=0)),  # b, 8, 12, 12
                ('17_d_act2', nn.ReLU(True)),
                ('18_d_convt3',
                 nn.ConvTranspose2d(4, 1, 10, stride=2,
                                    padding=0)),  # b, 1, 32, 32
                ('19_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_TA11_Nopad_32_R_0_Small_filters_Inc(nn.Module):
    def __init__(self, dropout=0.0):
        super(autoencoder_TA11_Nopad_32_R_0_Small_filters_Inc, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'TA11_Nopad_32_R_0_Small_filters_Inc'
        self.dropout = nn.Dropout2d(dropout)
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 8, 5, stride=3,
                                        padding=0)),  # b, 64, 124, 124
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_conv2', nn.Conv2d(8, 16, 5, stride=1,
                                        padding=0)),  # b, 64, 60, 60
                ('4_e_act2', nn.ReLU(True)),
                ('5_e_conv3', nn.Conv2d(16, 32, 5, stride=3,
                                        padding=0)),  # b, 64, 28, 28
                ('6_e_act3', nn.ReLU(True)),
                ('7_e_conv4', nn.Conv2d(32, 64, 5, stride=1,
                                        padding=0)),  # b, 64, 12, 12
                ('8_e_act4', nn.ReLU(True)),
                ('9_e_maxpool4', nn.MaxPool2d(3, stride=1)),  # b, 64, 6, 6
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('10_d_convt1', nn.ConvTranspose2d(64, 32, 5,
                                                   stride=1)),  # b, 64, 10, 10
                ('11_d_act1', nn.ReLU(True)),
                ('12_d_convt2', nn.ConvTranspose2d(32,
                                                   16,
                                                   5,
                                                   stride=3,
                                                   padding=0)),  # b, 8, 12, 12
                ('13_d_act2', nn.ReLU(True)),
                ('14_d_convt3',
                 nn.ConvTranspose2d(16, 8, 5, stride=3,
                                    padding=0)),  # b, 1, 32, 32
                ('15_d_act2', nn.ReLU(True)),
                ('16_d_convt3',
                 nn.Conv2d(8, 1, 5, stride=3,
                           padding=0)),  # b, 1, 32, 32
                ('17_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_TA11_Nopad_32_R_1_Small_filters_Inc(nn.Module):
    def __init__(self, dropout=0.0):
        super(autoencoder_TA11_Nopad_32_R_1_Small_filters_Inc, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'TA11_Nopad_32_R_1_Small_filters_Inc'
        self.dropout = nn.Dropout2d(dropout)
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 4, 5, stride=3,
                                        padding=0)),  # b, 64, 124, 124
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_conv2', nn.Conv2d(4, 8, 5, stride=1,
                                        padding=0)),  # b, 64, 60, 60
                ('4_e_act2', nn.ReLU(True)),
                ('dropout', nn.Dropout2d(.25)),
                ('5_e_conv3', nn.Conv2d(8, 16, 5, stride=3,
                                        padding=0)),  # b, 64, 28, 28
                ('6_e_act3', nn.ReLU(True)),
                ('7_e_conv4', nn.Conv2d(16, 32, 5, stride=1,
                                        padding=0)),  # b, 64, 12, 12
                ('8_e_act4', nn.ReLU(True)),
                ('9_e_maxpool4', nn.MaxPool2d(3, stride=1)),  # b, 64, 6, 6
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('10_d_convt1', nn.ConvTranspose2d(32, 16, 5,
                                                   stride=1)),  # b, 64, 10, 10
                ('11_d_act1', nn.ReLU(True)),
                ('12_d_convt2', nn.ConvTranspose2d(16,
                                                   8,
                                                   5,
                                                   stride=3,
                                                   padding=0)),  # b, 8, 12, 12
                ('13_d_act2', nn.ReLU(True)),
                ('14_d_convt3',
                 nn.ConvTranspose2d(8, 4, 5, stride=3,
                                    padding=0)),  # b, 1, 32, 32
                ('15_d_act2', nn.ReLU(True)),
                ('16_d_convt3',
                 nn.Conv2d(4, 1, 5, stride=3,
                           padding=0)),  # b, 1, 32, 32
                ('17_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_TA11_Nopad_32_R_1_Small_filters_Inc_avgpool(nn.Module):
    def __init__(self, dropout=0.0):
        super(autoencoder_TA11_Nopad_32_R_1_Small_filters_Inc_avgpool, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'TA11_Nopad_32_R_1_Small_filters_Inc_avgpool'
        self.dropout = nn.Dropout2d(dropout)
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 4, 5, stride=3,
                                        padding=0)),  # b, 64, 124, 124
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_conv2', nn.Conv2d(4, 8, 5, stride=1,
                                        padding=0)),  # b, 64, 60, 60
                ('4_e_act2', nn.ReLU(True)),
                ('dropout', nn.Dropout2d(.25)),
                ('5_e_conv3', nn.Conv2d(8, 16, 5, stride=3,
                                        padding=0)),  # b, 64, 28, 28
                ('6_e_act3', nn.ReLU(True)),
                ('7_e_conv4', nn.Conv2d(16, 32, 5, stride=1,
                                        padding=0)),  # b, 64, 12, 12
                ('8_e_act4', nn.ReLU(True)),
                ('9_e_maxpool4', nn.AvgPool2d(3, stride=1)),  # b, 64, 6, 6
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('10_d_convt1', nn.ConvTranspose2d(32, 16, 5,
                                                   stride=1)),  # b, 64, 10, 10
                ('11_d_act1', nn.ReLU(True)),
                ('12_d_convt2', nn.ConvTranspose2d(16,
                                                   8,
                                                   5,
                                                   stride=3,
                                                   padding=0)),  # b, 8, 12, 12
                ('13_d_act2', nn.ReLU(True)),
                ('14_d_convt3',
                 nn.ConvTranspose2d(8, 4, 5, stride=3,
                                    padding=0)),  # b, 1, 32, 32
                ('15_d_act2', nn.ReLU(True)),
                ('16_d_convt3',
                 nn.Conv2d(4, 1, 5, stride=3,
                           padding=0)),  # b, 1, 32, 32
                ('17_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_TA11_Nopad_32_R_2_Small_filters_Inc(nn.Module):
    def __init__(self, dropout=0.0):
        super(autoencoder_TA11_Nopad_32_R_2_Small_filters_Inc, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'TA11_Nopad_32_R_2_Small_filters_Inc'
        self.dropout = nn.Dropout2d(dropout)
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(3, 4, 5, stride=3,
                                        padding=0)),  # b, 64, 124, 124
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_conv2', nn.Conv2d(4, 8, 5, stride=1,
                                        padding=0)),  # b, 64, 60, 60
                ('4_e_act2', nn.ReLU(True)),
                ('dropout', nn.Dropout2d(.25)),
                ('5_e_conv3', nn.Conv2d(8, 16, 5, stride=3,
                                        padding=0)),  # b, 64, 28, 28
                ('6_e_act3', nn.ReLU(True)),
                ('7_e_conv4', nn.Conv2d(16, 32, 5, stride=1,
                                        padding=0)),  # b, 64, 12, 12
                ('8_e_act4', nn.ReLU(True)),
                ('9_e_maxpool4', nn.MaxPool2d(3, stride=1)),  # b, 64, 6, 6
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('10_d_convt1', nn.ConvTranspose2d(32, 16, 5,
                                                   stride=1)),  # b, 64, 10, 10
                ('11_d_act1', nn.ReLU(True)),
                ('12_d_convt2', nn.ConvTranspose2d(16,
                                                   8,
                                                   5,
                                                   stride=3,
                                                   padding=0)),  # b, 8, 12, 12
                ('13_d_act2', nn.ReLU(True)),
                ('14_d_convt3',
                 nn.ConvTranspose2d(8, 4, 5, stride=3,
                                    padding=0)),  # b, 1, 32, 32
                ('15_d_act2', nn.ReLU(True)),
                ('16_d_convt3',
                 nn.Conv2d(4, 1, 5, stride=3,
                           padding=0)),  # b, 1, 32, 32
                ('17_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1
# Correction autoencoder


class autoencoder_TA10_R_Nopad_32_0_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_TA10_R_Nopad_32_0_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'TA10_R_Nopad_32_0_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 16, 8, stride=8,
                                        padding=0)),  # b, 64, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
                ('4_e_conv2', nn.Conv2d(16, 32, 3, stride=1,
                                        padding=0)),  # b, 16, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d', nn.ConvTranspose2d(32, 16, 5,
                                           stride=1)),  # b, 64, 10, 10
                ('8_d', nn.ReLU(True)),
                ('9_d', nn.ConvTranspose2d(16,
                                           16,
                                           5,
                                           stride=1,
                                           padding=0)),  # b, 8, 12, 12
                ('10_d', nn.ReLU(True)),
                ('11_d', nn.Conv2d(16, 16, 3, stride=1,
                                   padding=0)),  # b, 64, 16, 16
                ('12_d', nn.ReLU(True)),

                ('13_d', nn.ConvTranspose2d(16, 8, 5,
                                            stride=1)),  # b, 64, 10, 10
                ('14_d', nn.ReLU(True)),
                ('15_d', nn.ConvTranspose2d(8,
                                            8,
                                            5,
                                            stride=1,
                                            padding=0)),  # b, 8, 12, 12
                ('16_d', nn.ReLU(True)),
                ('17_e', nn.Conv2d(8, 8, 3, stride=1,
                                   padding=0)),  # b, 64, 16, 16
                ('18_d', nn.ReLU(True)),

                ('19_d', nn.ConvTranspose2d(8, 4, 5,
                                            stride=1)),  # b, 64, 10, 10
                ('20_d', nn.ReLU(True)),
                ('21_d', nn.ConvTranspose2d(4,
                                            4,
                                            5,
                                            stride=1,
                                            padding=0)),  # b, 8, 12, 12
                ('22_d', nn.ReLU(True)),
                ('23_e', nn.Conv2d(4, 4, 3, stride=1,
                                   padding=0)),  # b, 64, 16, 16
                ('24_d', nn.ReLU(True)),

                ('25_d', nn.ConvTranspose2d(4, 2, 5,
                                            stride=1)),  # b, 64, 10, 10
                ('26_d', nn.ReLU(True)),
                ('27_d', nn.ConvTranspose2d(2,
                                            2,
                                            5,
                                            stride=1,
                                            padding=0)),  # b, 8, 12, 12
                ('28_d', nn.ReLU(True)),
                ('29_e', nn.Conv2d(2, 2, 3, stride=1,
                                   padding=0)),  # b, 64, 16, 16
                ('30_d', nn.ReLU(True)),

                ('31_d', nn.ConvTranspose2d(2,
                                            1,
                                            4,
                                            stride=1,
                                            padding=0)),  # b, 8, 12, 12
                ('32_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_TA12_R_Nopad_32_0_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_TA12_R_Nopad_32_0_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'TA12_R_Nopad_32_0_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 16, 8, stride=8,
                                        padding=0)),  # b, 64, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
                ('4_e_conv2', nn.Conv2d(16, 32, 3, stride=1,
                                        padding=0)),  # b, 16, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d', nn.ConvTranspose2d(32, 16, 5,
                                           stride=1)),  # b, 64, 10, 10
                ('8_d', nn.ReLU(True)),
                ('9_d', nn.ConvTranspose2d(16,
                                           16,
                                           5,
                                           stride=1,
                                           padding=0)),  # b, 8, 12, 12
                ('10_d', nn.ReLU(True)),
                ('11_d', nn.Conv2d(16, 16, 3, stride=1,
                                   padding=0)),  # b, 64, 16, 16
                ('12_d', nn.ReLU(True)),

                ('13_d_upscale',
                 Upsample(scale_factor=2, mode='bilinear', align_corners=True)),

                ('14_d', nn.ConvTranspose2d(16, 4, 5,
                                            stride=1)),  # b, 64, 10, 10
                ('15_d', nn.ReLU(True)),
                ('16_d', nn.ConvTranspose2d(4,
                                            4,
                                            5,
                                            stride=1,
                                            padding=0)),  # b, 8, 12, 12
                ('17_d', nn.ReLU(True)),
                ('18_e', nn.Conv2d(4, 4, 3, stride=1,
                                   padding=0)),  # b, 64, 16, 16
                ('19_d', nn.ReLU(True)),

                ('20_d', nn.ConvTranspose2d(4,
                                            1,
                                            5,
                                            stride=1,
                                            padding=0)),  # b, 8, 12, 12
                ('21_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_TA15_R_Nopad_32_0_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_TA15_R_Nopad_32_0_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'TA15_R_Nopad_32_0_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 16, 8, stride=8,
                                        padding=0)),  # b, 64, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
                ('4_e_conv2', nn.Conv2d(16, 64, 3, stride=1,
                                        padding=0)),  # b, 16, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d', nn.ConvTranspose2d(64, 32, 5,
                                           stride=1)),
                ('8_d', nn.ReLU(True)),
                ('9_d', nn.ConvTranspose2d(32,
                                           16,
                                           5,
                                           stride=1,
                                           padding=0)),
                ('10_d', nn.ReLU(True)),

                ('13_d_upscale',
                 Upsample(scale_factor=2, mode='bilinear', align_corners=True)),

                ('14_d', nn.ConvTranspose2d(16, 8, 5,
                                            stride=1)),
                ('15_d', nn.ReLU(True)),
                ('16_d', nn.ConvTranspose2d(8, 1, 3,
                                            stride=1,
                                            padding=0)),
                ('17_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_TA15_R_Nopad_32_0_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_TA15_R_Nopad_32_0_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'TA15_R_Nopad_32_0_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 16, 8, stride=8,
                                        padding=0)),  # b, 64, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
                ('4_e_conv2', nn.Conv2d(16, 64, 3, stride=1,
                                        padding=0)),  # b, 16, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d', nn.ConvTranspose2d(64, 32, 5,
                                           stride=1)),
                ('8_d', nn.ReLU(True)),
                ('9_d', nn.ConvTranspose2d(32,
                                           16,
                                           5,
                                           stride=1,
                                           padding=0)),
                ('10_d', nn.ReLU(True)),

                ('13_d_upscale',
                 Upsample(scale_factor=2, mode='bilinear', align_corners=True)),

                ('14_d', nn.ConvTranspose2d(16, 8, 5,
                                            stride=1)),
                ('15_d', nn.ReLU(True)),
                ('16_d', nn.ConvTranspose2d(8, 1, 3,
                                            stride=1,
                                            padding=0)),
                ('17_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_TA15_R_Nopad_32_1_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_TA15_R_Nopad_32_1_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'TA15_R_Nopad_32_1_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 16, 8, stride=8,
                                        padding=0)),  # b, 64, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
                ('4_e_conv2', nn.Conv2d(16, 32, 3, stride=1,
                                        padding=0)),  # b, 16, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d', nn.ConvTranspose2d(32, 16, 5,
                                           stride=1)),
                ('8_d', nn.ReLU(True)),
                ('9_d', nn.ConvTranspose2d(16,
                                           8,
                                           5,
                                           stride=1,
                                           padding=0)),
                ('10_d', nn.ReLU(True)),

                ('13_d_upscale',
                 Upsample(scale_factor=2, mode='bilinear', align_corners=True)),

                ('14_d', nn.ConvTranspose2d(8, 4, 5,
                                            stride=1)),
                ('15_d', nn.ReLU(True)),
                ('16_d', nn.ConvTranspose2d(4, 1, 3,
                                            stride=1,
                                            padding=0)),
                ('17_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_TA16_R_Nopad_32_0_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_TA16_R_Nopad_32_0_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'TA16_R_Nopad_32_0_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 16, 8, stride=8,
                                        padding=0)),  # b, 64, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
                ('4_e_conv2', nn.Conv2d(16, 32, 3, stride=1,
                                        padding=0)),  # b, 16, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d', nn.ConvTranspose2d(32, 24, 5,
                                           stride=1)),
                ('8_d', nn.ReLU(True)),
                ('9_d', nn.ConvTranspose2d(24,
                                           16,
                                           5,
                                           stride=1,
                                           padding=0)),
                ('10_d', nn.ReLU(True)),
                ('11_d', nn.ConvTranspose2d(16,
                                            8,
                                            5,
                                            stride=1,
                                            padding=0)),
                ('12_d', nn.ReLU(True)),
                ('13_d', nn.ConvTranspose2d(8,
                                            4,
                                            5,
                                            stride=1,
                                            padding=0)),
                ('14_d', nn.ReLU(True)),
                ('15_d', nn.ConvTranspose2d(4, 2, 5,
                                            stride=1)),
                ('16_d', nn.ReLU(True)),
                ('17_d', nn.ConvTranspose2d(2, 2, 5,
                                            stride=1)),
                ('18_d', nn.ReLU(True)),
                ('19_d', nn.ConvTranspose2d(2, 1, 4,
                                            stride=1,
                                            padding=0)),
                ('20_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_TA17_R_Nopad_32_0_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_TA17_R_Nopad_32_0_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'TA17_R_Nopad_32_0_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 16, 8, stride=8,
                                        padding=0)),  # b, 64, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
                ('4_e_conv2', nn.Conv2d(16, 32, 3, stride=1,
                                        padding=0)),  # b, 16, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d', nn.ConvTranspose2d(32, 16, 7,
                                           stride=1)),
                ('8_d', nn.ReLU(True)),
                ('9_d', nn.ConvTranspose2d(16, 8, 5,
                                           stride=1,
                                           padding=0)),
                ('10_d', nn.ReLU(True)),
                ('11_d', nn.ConvTranspose2d(8, 4, 7,
                                            stride=1,
                                            padding=0)),
                ('12_d', nn.ReLU(True)),
                ('13_d', nn.ConvTranspose2d(4, 2, 5,
                                            stride=1,
                                            padding=0)),
                ('14_d', nn.ReLU(True)),
                ('15_d', nn.ConvTranspose2d(2, 1, 8,
                                            stride=1)),
                ('16_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_TA_2_17_R_Nopad_32_0_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_TA_2_17_R_Nopad_32_0_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'TA_2_17_R_Nopad_32_0_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 16, 8, stride=8,
                                        padding=0)),  # b, 64, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
                ('4_e_conv2', nn.Conv2d(16, 32, 3, stride=1,
                                        padding=0)),  # b, 16, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d', nn.ConvTranspose2d(32, 16, 5,
                                           stride=1)),
                ('8_d', nn.ReLU(True)),
                ('9_d', nn.ConvTranspose2d(16, 8, 5,
                                           stride=1,
                                           padding=0)),
                ('10_d', nn.ReLU(True)),
                ('11_d', nn.ConvTranspose2d(8, 4, 7,
                                            stride=1,
                                            padding=0)),
                ('12_d', nn.ReLU(True)),
                ('13_d', nn.ConvTranspose2d(4, 2, 7,
                                            stride=1,
                                            padding=0)),
                ('14_d', nn.ReLU(True)),
                ('15_d', nn.ConvTranspose2d(2, 1, 8,
                                            stride=1)),
                ('16_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_TA_3_17_R_Nopad_32_0_Dense_Dec(nn.Module):
    def __init__(self):
        super(autoencoder_TA_3_17_R_Nopad_32_0_Dense_Dec, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'TA_3_17_R_Nopad_32_0_Dense_Dec'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv1', nn.Conv2d(4, 16, 8, stride=8,
                                        padding=0)),  # b, 64, 16, 16
                ('2_e_act1', nn.ReLU(True)),
                ('3_e_maxpool1', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
                ('4_e_conv2', nn.Conv2d(16, 32, 3, stride=1,
                                        padding=0)),  # b, 16, 6, 6
                ('5_e_act2', nn.ReLU(True)),
                ('6_e_maxpool2', nn.MaxPool2d(2, stride=1))  # b, 16, 5, 5
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('7_d', nn.ConvTranspose2d(32, 16, 8,
                                           stride=1)),
                ('8_d', nn.ReLU(True)),
                ('9_d', nn.ConvTranspose2d(16, 8, 7,
                                           stride=1,
                                           padding=0)),
                ('10_d', nn.ReLU(True)),
                ('11_d', nn.ConvTranspose2d(8, 4, 7,
                                            stride=1,
                                            padding=0)),
                ('12_d', nn.ReLU(True)),
                ('13_d', nn.ConvTranspose2d(4, 2, 5,
                                            stride=1,
                                            padding=0)),
                ('14_d', nn.ReLU(True)),
                ('15_d', nn.ConvTranspose2d(2, 1, 5,
                                            stride=1)),
                ('16_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_BA1_32_0(nn.Module):
    def __init__(self):
        super(autoencoder_BA1_32_0, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'BA1_32_0'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv', nn.Conv2d(4, 64, 8, stride=8,
                                       padding=0)),  # b, 64, 16, 16
                ('2_e_act', nn.ReLU(True)),
                ('3_e_maxpool', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('4_d_convt', nn.ConvTranspose2d(64, 32, 3,
                                                 stride=1)),
                ('5_d_act', nn.ReLU(True)),
                ('6_d_convt', nn.ConvTranspose2d(32, 16, 3,
                                                 stride=1)),
                ('7_d_act', nn.ReLU(True)),
                ('7_d_convt', nn.ConvTranspose2d(16, 1, 10,
                                                 stride=2)),
                ('8_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1


class autoencoder_BA3_32_0(nn.Module):
    def __init__(self):
        super(autoencoder_BA3_32_0, self).__init__()
        self.criterion = nn.MSELoss()
        self.case = 'BA3_32_0'
        self.encoder = nn.Sequential(
            OrderedDict([
                ('1_e_conv', nn.Conv2d(4, 64, 8, stride=8,
                                       padding=0)),  # b, 64, 16, 16
                ('2_e_act', nn.ReLU(True)),
                ('3_e_maxpool', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
                ('4_e_conv', nn.Conv2d(4, 64, 8, stride=8,
                                       padding=0)),  # b, 64, 16, 16
                ('5_e_act', nn.ReLU(True)),
                ('6_e_maxpool', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
                ('7_e_conv', nn.Conv2d(4, 64, 8, stride=8,
                                       padding=0)),  # b, 64, 16, 16
                ('8_e_act', nn.ReLU(True)),
                ('9_e_maxpool', nn.MaxPool2d(2, stride=2)),  # b, 64, 8, 8
            ]))

        self.decoder = nn.Sequential(
            OrderedDict([
                ('4_d_convt', nn.ConvTranspose2d(64, 32, 3,
                                                 stride=1)),
                ('5_d_act', nn.ReLU(True)),
                ('6_d_convt', nn.ConvTranspose2d(32, 16, 3,
                                                 stride=1)),
                ('7_d_act', nn.ReLU(True)),
                ('7_d_convt', nn.ConvTranspose2d(16, 1, 10,
                                                 stride=2)),
                ('8_out', nn.Tanh())
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, 1
