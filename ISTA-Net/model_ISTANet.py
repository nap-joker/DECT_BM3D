import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np 
import os
from numpy import *
import matplotlib.pyplot as plt
###########################################################################
# Define MRI-RDB Block
class BasicBlock(torch.nn.Module):   #定义了ISTANet中使用的基本模块
    def __init__(self,features):  #在init方法中，初始化了一系列参数，包括lambda_step(步长)、soft_thr（软阈值）、前向和后向的卷积核参数
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(features, 3, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(features, features, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(features, features, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(3, features, 3, 3)))

    def forward(self, x,sino_low,sino_high,primal_op_layer, dual_op_layer,
                dualEnergyTransmission,model_D,device, delta_e, modeldata_low,modeldata_high):
        # 在forward方法中，实现了前向传播过程。首先，根据输入数据进行一系列操作，包括计算正向投影、生成透射比、生成低能量和高能量的正向投影、进行滤波反投影重建、基材分解等
        #然后对输入数据进行卷积操作，通过激活函数RELU和软阈值函数实现特征提取和压缩，最终得到输出结果。
        x_A = x[:,0:1,:,:] #[1, 1, 512, 512]
        x_F = x[:,1:2,:,:]
        x_C = x[:,2:3,:,:]

        sinogram_odl_A = primal_op_layer(x_A) #[1, 1, 64, 1024]
        sinogram_odl_F = primal_op_layer(x_F)
        sinogram_odl_C = primal_op_layer(x_C)
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
        FBP_low = 2 * dual_op_layer(sinogram_low-sino_low) #[1, 1, 512, 512]
        FBP_high = 2 * dual_op_layer(sinogram_high-sino_high)

        FBP_trans = torch.cat((FBP_low, FBP_high), 1).to(device) #[1, 2, 512, 512]
        x_De = model_D(FBP_trans)  # 基材分解 #[1, 3, 512, 512]
        x_input = x - self.lambda_step * x_De
        
        x = F.conv2d(x_input, self.conv1_forward, padding=1) #[1, 32, 512, 512]
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1) #[1, 32, 512, 512]
        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr)) #[1, 32, 512, 512]
        x = F.conv2d(x, self.conv1_backward, padding=1) #[1, 32, 512, 512]
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1) #[1, 3, 512, 512]
        x_pred = x_backward

        x = F.conv2d(x_forward, self.conv1_backward, padding=1) #[1, 32, 512, 512]
        x = F.relu(x)
        x_est = F.conv2d(x, self.conv2_backward, padding=1) #[1, 3, 512, 512]
        symloss = x_est - x_input

        return [x_pred, symloss]
#####################################################################################################
# Define Deep Geometric Distillation Network
class ISTANet(torch.nn.Module):
    def __init__(self, LayerNo, num_feature):  #初始化ISTANet的参数，包括网络层数和特征数量，并构建了多个BasicBlock模块
        super(ISTANet, self).__init__()
        self.LayerNo = LayerNo

        onelayer = []
        for i in range(LayerNo):
            onelayer.append(BasicBlock(num_feature))

        self.fcs = nn.ModuleList(onelayer)

    def forward(self,Truth_A_T, Truth_F_T, Truth_C_T, primal_op_layer, dual_op_layer,
                dualEnergyTransmission,model_D,device, delta_e, modeldata_low,modeldata_high):
        #实现了整个网络的前向传播过程。首先，根据输入的数据进行一系列操作，包括计算正向投影、生成透射比、生成低能量和高能量的正向投影、进行滤波反投影重建等。
        #然后，通过循环遍历每个BasicBlock模块，逐层进行特征提取和压缩，并保存每一层的对称损失。最后，返回压缩后的特征以及每一层的对称损失
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
        x_D = model_D(FBP_transmission)  # 基材分解 #[1, 3, 512, 512]
        x = x_D
        layers_sym = []  # for computing symmetric loss
        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x,sinogram_low,sinogram_high,primal_op_layer, dual_op_layer,
                                         dualEnergyTransmission,model_D,device, delta_e, modeldata_low,modeldata_high)
            layers_sym.append(layer_sym)
        xnew = x #[1, 3, 512, 512]
        return [xnew, layers_sym, layers_sym]
