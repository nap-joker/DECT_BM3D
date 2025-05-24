import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np 
import os
from numpy import *
import matplotlib.pyplot as plt

@torch.no_grad()
def init_weights(init_type='xavier'):
    if init_type == 'xavier':
        init = nn.init.xavier_normal_
    elif init_type == 'he':
        init = nn.init.kaiming_normal_
    else:
        init = nn.init.orthogonal_

    def initializer(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            init(m.weight)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.01)
            nn.init.zeros_(m.bias)

    return initializer


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.actv = nn.PReLU(out_channels)

    def forward(self, x):
        return self.actv(self.conv(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, cat_channels, out_channels):
        super(UpsampleBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels + cat_channels, out_channels, 3, padding=1)
        self.conv_t = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.actv = nn.PReLU(out_channels)
        self.actv_t = nn.PReLU(in_channels)

    def forward(self, x):
        upsample, concat = x
        upsample = self.actv_t(self.conv_t(upsample))
        return self.actv(self.conv(torch.cat([concat, upsample], 1)))


class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InputBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.actv_1 = nn.PReLU(out_channels)
        self.actv_2 = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.actv_1(self.conv_1(x))
        return self.actv_2(self.conv_2(x))


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutputBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.actv_1 = nn.PReLU(in_channels)
        self.actv_2 = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.actv_1(self.conv_1(x))
        return self.actv_2(self.conv_2(x))


class DenoisingBlock(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels):
        super(DenoisingBlock, self).__init__()
        self.conv_0 = nn.Conv2d(in_channels, inner_channels, 3, padding=1)
        self.conv_1 = nn.Conv2d(in_channels + inner_channels, inner_channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels + 2 * inner_channels, inner_channels, 3, padding=1)
        self.conv_3 = nn.Conv2d(in_channels + 3 * inner_channels, out_channels, 3, padding=1)

        self.actv_0 = nn.PReLU(inner_channels)
        self.actv_1 = nn.PReLU(inner_channels)
        self.actv_2 = nn.PReLU(inner_channels)
        self.actv_3 = nn.PReLU(out_channels)

    def forward(self, x):
        out_0 = self.actv_0(self.conv_0(x))

        out_0 = torch.cat([x, out_0], 1)
        out_1 = self.actv_1(self.conv_1(out_0))

        out_1 = torch.cat([out_0, out_1], 1)
        out_2 = self.actv_2(self.conv_2(out_1))

        out_2 = torch.cat([out_1, out_2], 1)
        out_3 = self.actv_3(self.conv_3(out_2))

        return out_3 + x


class RDUNet(nn.Module):
    r"""
    Residual-Dense U-net for image denoising.
    """
    def __init__(self, **kwargs):
        super().__init__()

        channels = 1
        filters_0 = 512
        filters_1 = 2 * filters_0
        filters_2 = 4 * filters_0
        filters_3 = 8 * filters_0

        # Encoder:
        # Level 0:
        self.input_block = InputBlock(channels, filters_0)
        self.block_0_0 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.block_0_1 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.down_0 = DownsampleBlock(filters_0, filters_1)

        # Level 1:
        self.block_1_0 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.block_1_1 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.down_1 = DownsampleBlock(filters_1, filters_2)

        # Level 2:
        self.block_2_0 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.block_2_1 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.down_2 = DownsampleBlock(filters_2, filters_3)

        # Level 3 (Bottleneck)
        self.block_3_0 = DenoisingBlock(filters_3, filters_3 // 2, filters_3)
        self.block_3_1 = DenoisingBlock(filters_3, filters_3 // 2, filters_3)

        # Decoder
        # Level 2:
        self.up_2 = UpsampleBlock(filters_3, filters_2, filters_2)
        self.block_2_2 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.block_2_3 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)

        # Level 1:
        self.up_1 = UpsampleBlock(filters_2, filters_1, filters_1)
        self.block_1_2 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.block_1_3 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)

        # Level 0:
        self.up_0 = UpsampleBlock(filters_1, filters_0, filters_0)
        self.block_0_2 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.block_0_3 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)

        self.output_block = OutputBlock(filters_0, channels)

    def forward(self, Truth_A_T, Truth_F_T, Truth_C_T, primal_op_layer, dual_op_layer,
                dualEnergyTransmission,model_D,device, delta_e, modeldata_low,modeldata_high):
        x_truth = torch.cat([Truth_A_T, Truth_F_T, Truth_C_T], dim=1)
        sinogram_odl_A = primal_op_layer(Truth_A_T) #[1, 1, 64, 1024]
        sinogram_odl_F = primal_op_layer(Truth_F_T)
        sinogram_odl_C = primal_op_layer(Truth_C_T)

        transmission_low = dualEnergyTransmission(sinogram_odl_A[:, :, 1::2, :], sinogram_odl_F[:, :, 1::2, :],
                                                  sinogram_odl_C[:, :, 1::2, :], delta_e, modeldata_low,
                                                  device) #[1, 1, 32, 1024]

        transmission_high = dualEnergyTransmission(sinogram_odl_A[:, :, ::2, :], sinogram_odl_F[:, :, ::2, :],
                                                   sinogram_odl_C[:, :, ::2, :], delta_e, modeldata_high,
                                                   device)

        sinogram_low = torch.zeros([transmission_low.shape[0], transmission_low.shape[1],
                                    2 * transmission_low.shape[2], transmission_low.shape[3]],
                                   dtype=torch.float).to(device)
        sinogram_low[:, :, 1::2, :] = -torch.log(transmission_low) #[1, 1, 64, 1024]
        sinogram_high = torch.zeros([transmission_low.shape[0], transmission_low.shape[1],
                                     2 * transmission_low.shape[2], transmission_low.shape[3]],
                                    dtype=torch.float).to(device)
        sinogram_high[:, :, 0::2, :] = -torch.log(transmission_high)

        FBP_low = 2 * dual_op_layer(sinogram_low) #[1, 1, 512, 512]
        FBP_high = 2 * dual_op_layer(sinogram_high)

        FBP_transmission = torch.cat((FBP_low, FBP_high), 1).to(device) #[1, 2, 512, 512]
        x_D = model_D(FBP_transmission)  # »ù²Ä·Ö½â #[1, 3, 512, 512]
        inputs = x_D
        
        out_0 = self.input_block(inputs)    # Level 0
        out_0 = self.block_0_0(out_0)
        out_0 = self.block_0_1(out_0)

        out_1 = self.down_0(out_0)          # Level 1
        out_1 = self.block_1_0(out_1)
        out_1 = self.block_1_1(out_1)

        out_2 = self.down_1(out_1)          # Level 2
        out_2 = self.block_2_0(out_2)
        out_2 = self.block_2_1(out_2)

        out_3 = self.down_2(out_2)          # Level 3 (Bottleneck)
        out_3 = self.block_3_0(out_3)
        out_3 = self.block_3_1(out_3)

        out_4 = self.up_2([out_3, out_2])   # Level 2
        out_4 = self.block_2_2(out_4)
        out_4 = self.block_2_3(out_4)

        out_5 = self.up_1([out_4, out_1])   # Level 1
        out_5 = self.block_1_2(out_5)
        out_5 = self.block_1_3(out_5)

        out_6 = self.up_0([out_5, out_0])   # Level 0
        out_6 = self.block_0_2(out_6)
        out_6 = self.block_0_3(out_6)
        
        x_new = self.output_block(out_6) + inputs
        est_loss = x_new - x_truth
        return [x_new,est_loss,est_loss]
