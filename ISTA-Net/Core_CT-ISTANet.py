#用于对医学图像进行重建的深度学习模型的训练和测试脚本
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  #指定了使用的GPU设备
import torch
import torch.nn as nn
import numpy as np
from argparse import ArgumentParser  #使用其定义一系列命令行参数，用于指定模型、训练参数和数据路径等
# import torchvision
from data_loader import *
from model_ISTANet import ISTANet
from model_decomposition import D_net
from solver_CT import Solver_CT
###########################################################################################
# parameter
parser = ArgumentParser(description='Dual-CT-new')
parser.add_argument('--net_name', type=str, default='Dual-CT-ISTANet', help='name of net')  #网络名称，用于指定模型的名称
parser.add_argument('--net_name_D', type=str, default='Dual-CT-decomposition', help='name of net')  #基材分解网络的名称
parser.add_argument('--model_dir', type=str, default='model_spectral_CT', help='model_MRI,model_CT,trained or pre-trained model directory')   #模型保存目录，指定训练或预训练模型的存储路径
parser.add_argument('--model_dir_D', type=str, default='model_spectral_CT_decomposition', help='model_MRI,model_CT,trained or pre-trained model directory')  #基材分解模型的保存目录
parser.add_argument('--log_dir', type=str, default='log_spectral_CT', help='log_MRI,log_CT')   #日志目录，用于保存训练过程中的日志信息
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')   #起始训练轮数，指定训练从哪个epoch开始
parser.add_argument('--end_epoch', type=int, default=500, help='epoch number of end training')   #结束训练轮数，指定训练在哪个epoch结束
parser.add_argument('--start_epoch_D', type=int, default=200, help='epoch number of start training')   #基材分解模型训练的起始轮数
parser.add_argument('--batch_size', type=int, default=1, help='MRI=1,CT=1')   #批量大小，指定训练时每个批次的样本数量
parser.add_argument('--ds_factor', type=int, default=16, help='{16, 8} for fan-beam')   #下采样因子，用于指定图像的下采样因子
parser.add_argument('--layer_num', type=int, default=7, help='D,11-7.5G')   #模型的层数
parser.add_argument('--num_features', type=int, default=32, help='G,32')   #模型的特征数量
# parser.add_argument('--num_layers', type=int, default=4, help='C,6,8')
parser.add_argument('--num_features_D', type=int, default=32, help='G,64')   #基材分解模型的特征数量
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')   #学习率，指定模型训练时的学习率
parser.add_argument('--lr_D', type=float, default=1e-4, help='learning rate')   #基材分解模型的学习率
parser.add_argument('--num_work', type=int, default=4, help='4,1')   #工作进程数量，用于指定数据加载时的工作进程数量
parser.add_argument('--print_flag', type=int, default=1, help='print parameter number 1 or 0')   #是否打印网络参数数量的标志，1表示打印，0表示不打印
parser.add_argument('--result_dir', type=str, default='result', help='result directory')   #结果目录，指定测试结果的保存路径
parser.add_argument('--CT_test_name', type=str, default='Spectral_CT_test', help='name of CT test set')   #CT测试数据集的名称
parser.add_argument('--run_mode', type=str, default='train', help='train、test')    #运行模式，可以是train或者test，指定是进行模型训练还是模型测试
parser.add_argument('--save_image_mode', type=int, default=1, help='save 1, not 0 in test')   #保存图片模式，在测试时用于指定是否保存图像，1表示保存，0表述不保存
parser.add_argument('--loss_mode', type=str, default='Ista-net', help='vggloss,midloss,L1,L2,Fista-net')   #损失函数模式，用于指定训练时使用的损失函数
parser.add_argument('--penalty_mode', type=str, default='None', help='Fista-net,Weight,None')   #惩罚模式，用于指定基材分解模型的惩罚方式
args = parser.parse_args()
#########################################################################################
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
###########################################################################################
# data loading
train_re_d, val_re_d, test_re_d ,train_de_d, val_de_d =  CT_dataloader(args.num_work,args.batch_size)  #使用CT_dataloader函数加载训练、验证和测试数据集
###################################################################################
# model
model = ISTANet(args.layer_num, args.num_features)  #ISTA-NET是一个神经网络模型，用于对医学图像进行重建
model = nn.DataParallel(model)
model = model.to(device)

# 加载基材分解模型
model_D = D_net(args.num_features_D)  #D_NET是一个基材分解模型，用于对医学图像进行基材分解
model_D = nn.DataParallel(model_D)
model_D = model_D.to(device)
model_dir_D = "%s/%s_lr_%f" % (args.model_dir_D, args.net_name_D, args.lr_D)
if torch.cuda.is_available():
    model_D.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir_D, args.start_epoch_D)))
else:
    model_D.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir_D, args.start_epoch_D), map_location=torch.device('cpu')))

model_D.to(device)
###################################################################################
if args.print_flag:  # print networks parameter number
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
#######################################################################################
if args.run_mode == 'train':
    solver = Solver_CT(model,model_D, train_re_d, val_re_d, test_re_d, args, device)
    solver.train()
elif args.run_mode == 'test':
    solver = Solver_CT(model,model_D, train_re_d, val_re_d ,test_re_d, args, device)
    solver.test()
#########################################################################################
