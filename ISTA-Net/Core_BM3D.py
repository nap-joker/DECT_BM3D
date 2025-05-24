import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn as nn
import numpy as np
from argparse import ArgumentParser
# import torchvision
from data_loader import *
from model_BM3D import BM3D
from model_decomposition import D_net
from solver_CT_BM3D import Solver_CT

parser = ArgumentParser(description='Dual-CT-new')
parser.add_argument('--net_name', type=str, default='Dual-CT-BM3D', help='name of net')
parser.add_argument('--net_name_D', type=str, default='Dual-CT-decomposition', help='name of net')
parser.add_argument('--model_dir', type=str, default='model_spectral_CT', help='model_MRI,model_CT,trained or pre-trained model directory')
parser.add_argument('--model_dir_D', type=str, default='model_spectral_CT_decomposition', help='model_MRI,model_CT,trained or pre-trained model directory')
parser.add_argument('--log_dir', type=str, default='log_spectral_CT', help='log_MRI,log_CT')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=500, help='epoch number of end training')
parser.add_argument('--start_epoch_D', type=int, default=200, help='epoch number of start training')
parser.add_argument('--batch_size', type=int, default=1, help='MRI=1,CT=1')
parser.add_argument('--ds_factor', type=int, default=16, help='{16, 8} for fan-beam')
parser.add_argument('--num_features_D', type=int, default=32, help='G,64')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_D', type=float, default=1e-4, help='learning rate')
parser.add_argument('--num_work', type=int, default=4, help='4,1')
parser.add_argument('--print_flag', type=int, default=1, help='print parameter number 1 or 0')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--CT_test_name', type=str, default='Spectral_CT_test', help='name of CT test set')
parser.add_argument('--run_mode', type=str, default='test', help='train,test')
parser.add_argument('--save_image_mode', type=int, default=1, help='save 1, not 0 in test')
parser.add_argument('--loss_mode', type=str, default='BM3D', help='vggloss,midloss,L1,L2,Fista-net,ISTA-net')
parser.add_argument('--penalty_mode', type=str, default='None', help='Fista-net,Weight,None')
args = parser.parse_args()

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_re_d, val_re_d, test_re_d ,train_de_d, val_de_d =  CT_dataloader(args.num_work,args.batch_size)

model = BM3D(lamb2d=2.0,lamb3d=2.7)
model = nn.DataParallel(model)
model = model.to(device)

model_D = D_net(args.num_features_D)
model_D = nn.DataParallel(model_D)
model_D = model_D.to(device)
model_dir_D = "%s/%s_lr_%f" % (args.model_dir_D, args.net_name_D, args.lr_D)
if torch.cuda.is_available():
    model_D.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir_D, args.start_epoch_D)))
else:
    model_D.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir_D, args.start_epoch_D), map_location=torch.device('cpu')))

model_D.to(device)

if args.print_flag:
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))

if args.run_mode == 'train':
    solver = Solver_CT(model,model_D, train_re_d, val_re_d, test_re_d, args, device)
    solver.train()
elif args.run_mode == 'test':
    solver = Solver_CT(model,model_D, train_re_d, val_re_d ,test_re_d, args, device)
    solver.test()