import os
import torch
import torch.nn as nn
from functions import * #�Զ���ĺ���ģ��
import glob   #���ڴ��ļ����ж�ȡ�ļ��б�
#from skimage.measure import compare_ssim as ssim
from skimage.metrics import structural_similarity as ssim  #��skimage��ĸ��°汾�б���������
from functions import *
import cv2
from time import time
import matplotlib.pyplot as plt
import random
from numpy import *
import numpy as np
import odl  #Optimal design library ������ѧ��ģ���Ż�����
import odl.contrib.torch as odl_torch   #ODL���PyTorch�ļ���
###########################################################################
def dualEnergyTransmission(sa, sf, sc, delta_e, modeldata,device):   #�������������˫��CTͼ��Ĵ��䣬����һЩ�����������������ݣ�sa��sf��sc����������ֵ��ģ������
    nviews, nbins = sa.shape[2], sa.shape[3] # Grab the data dimensions from the Sinogram data
    delta_e = torch.tensor(delta_e).to(device)
    modeldata = torch.from_numpy(modeldata).to(device)
    energies = modeldata[0] * 1.  # don't really need this data for computation. Only need for plotting eg: plot(energies,spectrum)
    spectrum = modeldata[1] * 1.
    mu_adipose = modeldata[2] * 1.
    mu_fibroglandular = modeldata[3] * 1.
    mu_calcification = modeldata[4] * 1.
    transmission = torch.zeros([sa.shape[0],sa.shape[1],nviews, nbins],dtype=torch.float).to(device)
    for j in range(len(energies)):
        exp_tmp = -mu_adipose[j] * sa - mu_fibroglandular[j] * sf - mu_calcification[j] * sc
        # # ��ֹexp
        exp_tmp[exp_tmp>5] = 5  # ���������������ֵתΪ��ֵ
        # trans_tmp = torch.mul(delta_e, spectrum[j])
        # transmission = transmission + torch.mul(trans_tmp, exp(exp_tmp))
        transmission += delta_e * spectrum[j] * torch.exp(exp_tmp)
    return transmission   #��������Ĳ������㴫�䣬�����ش�������
###########################################################################
def projection_operator(ximageside,yimageside,Truth_A_T,slen,nviews,nbins,radius,source_to_detector):   #���ڴ���ͶӰ���ӣ���ԭʼͼ��ת��ΪͶӰ����
    # creat low/high FBP
    reco_space = odl.uniform_discr(min_pt=[-ximageside / 2, -yimageside / 2],
                                   max_pt=[ximageside / 2, yimageside / 2],
                                   shape=[Truth_A_T.shape[2], Truth_A_T.shape[2]], dtype='float32')
    angle_partition = odl.uniform_partition(0, slen, nviews)  # Make a fan beam geometry with flat detector
    detector_partition = odl.uniform_partition(-ximageside, ximageside, nbins)
    reco_geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition,
                                             src_radius=radius, det_radius=(source_to_detector - radius))
    reco_ray_trafo = odl.tomo.RayTransform(reco_space, reco_geometry, impl='astra_cuda')
    primal_op_layer = odl_torch.OperatorModule(reco_ray_trafo)
    reco_ray_trafo_fbp = odl.tomo.fbp_op(reco_ray_trafo)  # Create FBP operator using utility function
    dual_op_layer = odl_torch.OperatorModule(reco_ray_trafo_fbp)

    return primal_op_layer,dual_op_layer
###########################################################################
class Solver_CT(object):   #CTͼ���ؽ�����������
    def __init__(self, model,model_D, train_loader,val_loader, test_loader, args, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.lr = args.learning_rate
        self.start_epoch = args.start_epoch
        self.end_epoch = args.end_epoch
        self.device = device
        self.loss_mode = args.loss_mode
        self.run_mode = args.run_mode
        self.CT_test_name = args.CT_test_name
        self.model_dir = args.model_dir
        self.net_name = args.net_name
        self.ds_factor = args.ds_factor
        self.log_dir = args.log_dir
        self.result_dir = args.result_dir
        self.save_image_mode = args.save_image_mode
        self.end_epoch = args.end_epoch
        self.model_D = model_D

        self.ximageside = 18.0  # cm
        self.yimageside = 18.0  # cm
        self.radius = 50.0  # cm
        self.source_to_detector = 100.0  # cm
        self.nviews = int(1024/args.ds_factor)
        self.slen = 2 * pi  # angular range of the scan
        self.nbins = 1024  # number of detector pixels
        self.delta_e = 0.5
        self.kvp_low = "50"  # options are "50" (low) and "80" (high)
        self.kvp_high = "80"  # options are "50" (low) and "80" (high)
        self.modeldata_low = load("/20134138/AAPM2022/model_data_" + self.kvp_low + "kVp.npy")
        self.modeldata_high = load("/20134138/AAPM2022/model_data_" + self.kvp_high + "kVp.npy")

        # Only initialize optimizer for D_net if we're not using BM3D
        if args.loss_mode != 'BM3D':
            if args.penalty_mode == 'Fista-net':
                self.optimizer = torch.optim.Adam([
                    {'params': self.model.module.fcs.parameters()},
                    {'params': self.model.module.w_theta, 'lr': 0.0001},
                    {'params': self.model.module.b_theta, 'lr': 0.0001},
                    {'params': self.model.module.w_mu, 'lr': 0.0001},
                    {'params': self.model.module.b_mu, 'lr': 0.0001},
                    {'params': self.model.module.w_rho, 'lr': 0.0001},
                    {'params': self.model.module.b_rho, 'lr': 0.0001}],
                    lr=self.lr, weight_decay=0.0001)
            elif args.penalty_mode == 'None':
                self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

    def train(self):
        if self.loss_mode == 'BM3D':
            print("BM3D is a non-trainable model. Skipping training.")
            return
            
        # define save dir
        model_dir = "./%s/%s_lr_%f" % (
            self.model_dir, self.net_name, self.lr)
        # Load pre-trained model with epoch number
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        output_file = "./%s" % (self.log_dir)
        if not os.path.exists(output_file):
            os.makedirs(output_file)
        if self.start_epoch > 0:  # train stop and restart
            self.model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, self.start_epoch)))
            self.model.to(self.device)

        # Training loop
        best_RMSE = 1
        for epoch_i in range(self.start_epoch + 1, self.end_epoch + 1):
            self.model.train(True)
            step = 0
            for i, data in enumerate(self.train_loader):
                step = step + 1

                Truth_A_T, Truth_F_T, Truth_C_T = data
                Truth_A_T = torch.unsqueeze(Truth_A_T, 1).cpu().data.numpy()
                Truth_F_T = torch.unsqueeze(Truth_F_T, 1).cpu().data.numpy()
                Truth_C_T = torch.unsqueeze(Truth_C_T, 1).cpu().data.numpy()

                # data augment
                if random.random() < 0.5:  # random_horizontal_flip
                    Truth_A_T = Truth_A_T[:, :, :, ::-1].copy()
                    Truth_F_T = Truth_F_T[:, :, :, ::-1].copy()
                    Truth_C_T = Truth_C_T[:, :, :, ::-1].copy()

                if random.random() < 0.5:  # random_vertical_flip
                    Truth_A_T = Truth_A_T[:, :, ::-1, :].copy()
                    Truth_F_T = Truth_F_T[:, :, ::-1, :].copy()
                    Truth_C_T = Truth_C_T[:, :, ::-1, :].copy()

                # numpy to tensor
                Truth_A_T = torch.from_numpy(Truth_A_T).to(self.device)
                Truth_F_T = torch.from_numpy(Truth_F_T).to(self.device)
                Truth_C_T = torch.from_numpy(Truth_C_T).to(self.device)

                # creat low/high FBP
                primal_op_layer, dual_op_layer = \
                    projection_operator(self.ximageside, self.yimageside, Truth_A_T, self.slen, self.nviews,
                                        self.nbins, self.radius, self.source_to_detector)
                batch_x = torch.cat((Truth_A_T, Truth_F_T, Truth_C_T), 1).to(self.device)
                
                # predict
                [x_output,x_mid,x_last] = self.model(Truth_A_T, Truth_F_T, Truth_C_T,primal_op_layer,
                                                  dual_op_layer,dualEnergyTransmission,self.model_D,
                                                  self.device, self.delta_e, self.modeldata_low,
                                                  self.modeldata_high)

                if self.loss_mode == 'midloss':  # midloss
                    loss_discrepancy = torch.mean(torch.abs(x_output - batch_x))
                    loss_discrepancy_mid = torch.mean(torch.abs(x_mid - batch_x))
                    loss_all = loss_discrepancy + loss_discrepancy_mid

                elif self.loss_mode == 'L1':  # simple L1
                    loss_discrepancy = torch.mean(torch.abs(x_output - batch_x))
                    loss_all = loss_discrepancy

                elif self.loss_mode == 'L2':  # simple L1
                    train_loss = nn.MSELoss()
                    loss_discrepancy = train_loss(x_output, batch_x)
                    loss_all = loss_discrepancy

                elif self.loss_mode == 'Fista-net':  # simple L1
                    train_loss = nn.MSELoss()
                    loss_discrepancy = train_loss(x_output, batch_x)
                    loss_constraint = 0
                    for k, _ in enumerate(x_mid, 0):
                        loss_constraint += torch.mean(torch.pow(x_mid[k], 2))

                    encoder_constraint = 0
                    for k, _ in enumerate(x_last, 0):
                        encoder_constraint += torch.mean(torch.abs(x_last[k]))

                    loss_all = loss_discrepancy + 0.01 * loss_constraint + 0.001 * encoder_constraint

                elif self.loss_mode == 'Ista-net':  # simple L1
                    loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))

                    loss_constraint = torch.mean(torch.pow(x_mid[0], 2))
                    for k in range(self.layer_num - 1):
                        loss_constraint += torch.mean(torch.pow(x_mid[k + 1], 2))

                    gamma = torch.Tensor([0.01]).to(self.device)
                    loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint)

                # Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()
                loss_all.backward()
                self.optimizer.step()

                if step % 10 == 0:
                    output_data = "[%02d/%02d] Step:%.0f | Total Loss: %.6f | Discrepancy Loss: %.6f " % \
                                  (epoch_i, self.end_epoch, step, loss_all.item(), loss_discrepancy.item())
                    print(output_data)

            # val
            model = self.model.eval()
            with torch.no_grad():
                RMSE_A_total = []
                RMSE_F_total = []
                RMSE_C_total = []
                PSNR_A_total = []
                PSNR_F_total = []
                PSNR_C_total = []
                SSIM_A_total = []
                SSIM_F_total = []
                SSIM_C_total = []
                for i, data in enumerate(self.val_loader):
                    Truth_A_T, Truth_F_T, Truth_C_T = data
                    Truth_A_T = torch.unsqueeze(Truth_A_T, 1).to(self.device)
                    Truth_F_T = torch.unsqueeze(Truth_F_T, 1).to(self.device)
                    Truth_C_T = torch.unsqueeze(Truth_C_T, 1).to(self.device)

                    # predict
                    [x_output, x_mid, x_last] = model(Truth_A_T, Truth_F_T, Truth_C_T, primal_op_layer,
                                                   dual_op_layer, dualEnergyTransmission, self.model_D,
                                                   self.device, self.delta_e, self.modeldata_low,
                                                   self.modeldata_high)

                    # ������ָ��
                    Truth_A = torch.squeeze(Truth_A_T).cpu().data.numpy()
                    Truth_F = torch.squeeze(Truth_F_T).cpu().data.numpy()
                    Truth_C = torch.squeeze(Truth_C_T).cpu().data.numpy()
                    x_A = x_output[:,0:1, :, :]
                    x_F = x_output[:,1:2, :, :]
                    x_C = x_output[:,2:3, :, :]
                    x_A = torch.squeeze(x_A).cpu().data.numpy()
                    x_F = torch.squeeze(x_F).cpu().data.numpy()
                    x_C = torch.squeeze(x_C).cpu().data.numpy()

                    rec_A_RMSE = compute_measure(x_A, Truth_A, 1)
                    rec_F_RMSE = compute_measure(x_F, Truth_F, 1)
                    rec_C_RMSE = compute_measure(x_C, Truth_C, 1)

                    RMSE_A_total.append(rec_A_RMSE)
                    RMSE_F_total.append(rec_F_RMSE)
                    RMSE_C_total.append(rec_C_RMSE)

                    rec_A_PSNR = compute_measure(x_A, Truth_A, 2)
                    rec_F_PSNR = compute_measure(x_F, Truth_F, 2)
                    rec_C_PSNR = compute_measure(x_C, Truth_C, 2)

                    PSNR_A_total.append(rec_A_PSNR)
                    PSNR_F_total.append(rec_F_PSNR)
                    PSNR_C_total.append(rec_C_PSNR)

                    rec_A_SSIM = compute_measure(x_A, Truth_A, 3)
                    rec_F_SSIM = compute_measure(x_F, Truth_F, 3)
                    rec_C_SSIM = compute_measure(x_C, Truth_C, 3)

                    SSIM_A_total.append(rec_A_SSIM)
                    SSIM_F_total.append(rec_F_SSIM)
                    SSIM_C_total.append(rec_C_SSIM)

                RMSE_A_mean = np.mean(RMSE_A_total)
                RMSE_F_mean = np.mean(RMSE_F_total)
                RMSE_C_mean = np.mean(RMSE_C_total)

                PSNR_A_mean = np.mean(PSNR_A_total)
                PSNR_F_mean = np.mean(PSNR_F_total)
                PSNR_C_mean = np.mean(PSNR_C_total)

                SSIM_A_mean = np.mean(SSIM_A_total)
                SSIM_F_mean = np.mean(SSIM_F_total)
                SSIM_C_mean = np.mean(SSIM_C_total)

                output_data = "[%02d/%02d] | RMSE_A: %.6f | RMSE_F: %.6f | RMSE_C: %.6f | PSNR_A: %.6f | PSNR_F: %.6f | PSNR_C: %.6f | SSIM_A: %.6f | SSIM_F: %.6f | SSIM_C: %.6f " % \
                              (epoch_i, self.end_epoch, RMSE_A_mean, RMSE_F_mean, RMSE_C_mean, PSNR_A_mean, PSNR_F_mean, PSNR_C_mean, SSIM_A_mean, SSIM_F_mean, SSIM_C_mean)
                print(output_data)

                # save model
                if RMSE_A_mean < best_RMSE:
                    best_RMSE = RMSE_A_mean
                    torch.save(self.model.state_dict(), './%s/net_params_%d.pkl' % (model_dir, epoch_i))
                    print('model saved')

    def test(self):
        # Load pre-trained model with epoch number
        model_dir = "%s/%s_lr_%f" % (
            self.model_dir, self.net_name, self.lr)
        if self.loss_mode != 'BM3D':
            self.model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, self.end_epoch)))
        model = self.model.eval()

        result_dir_tmp = os.path.join(self.result_dir, self.CT_test_name)
        result_dir = result_dir_tmp + '_' + self.net_name + '_ds_' + str(self.ds_factor) + '_epoch_' + str(
            self.end_epoch) + '/'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        with torch.no_grad():
            RMSE_A_total = []
            RMSE_F_total = []
            RMSE_C_total = []
            PSNR_A_total = []
            PSNR_F_total = []
            PSNR_C_total = []
            SSIM_A_total = []
            SSIM_F_total = []
            SSIM_C_total = []
            image_num = 0
            for i, data in enumerate(self.test_loader):
                image_num = image_num + 1
                Truth_A_T, Truth_F_T, Truth_C_T = data
                Truth_A_T = torch.unsqueeze(Truth_A_T, 1).to(self.device)
                Truth_F_T = torch.unsqueeze(Truth_F_T, 1).to(self.device)
                Truth_C_T = torch.unsqueeze(Truth_C_T, 1).to(self.device)

                # predict
                if self.loss_mode == 'BM3D':
                    # Apply material decomposition first
                    FBP_trans = torch.cat((Truth_A_T, Truth_F_T), 1).to(self.device)
                    x_D = self.model_D(FBP_trans)  # Material decomposition
                    
                    # Apply BM3D to each channel
                    x_A = self.model(x_D[:, 0:1, :, :])
                    x_F = self.model(x_D[:, 1:2, :, :])
                    x_C = self.model(x_D[:, 2:3, :, :])
                    x_output = torch.cat((x_A, x_F, x_C), 1)
                else:
                    [x_output, x_mid, x_last] = model(Truth_A_T, Truth_F_T, Truth_C_T, primal_op_layer,
                                                   dual_op_layer, dualEnergyTransmission, self.model_D,
                                                   self.device, self.delta_e, self.modeldata_low,
                                                   self.modeldata_high)

                # ������ָ��
                Truth_A = torch.squeeze(Truth_A_T).cpu().data.numpy()
                Truth_F = torch.squeeze(Truth_F_T).cpu().data.numpy()
                Truth_C = torch.squeeze(Truth_C_T).cpu().data.numpy()
                x_A = x_output[:,0:1, :, :]
                x_F = x_output[:,1:2, :, :]
                x_C = x_output[:,2:3, :, :]
                x_A = torch.squeeze(x_A).cpu().data.numpy()
                x_F = torch.squeeze(x_F).cpu().data.numpy()
                x_C = torch.squeeze(x_C).cpu().data.numpy()

                rec_A_RMSE = compute_measure(x_A, Truth_A, 1)
                rec_F_RMSE = compute_measure(x_F, Truth_F, 1)
                rec_C_RMSE = compute_measure(x_C, Truth_C, 1)

                RMSE_A_total.append(rec_A_RMSE)
                RMSE_F_total.append(rec_F_RMSE)
                RMSE_C_total.append(rec_C_RMSE)

                rec_A_PSNR = compute_measure(x_A, Truth_A, 2)
                rec_F_PSNR = compute_measure(x_F, Truth_F, 2)
                rec_C_PSNR = compute_measure(x_C, Truth_C, 2)

                PSNR_A_total.append(rec_A_PSNR)
                PSNR_F_total.append(rec_F_PSNR)
                PSNR_C_total.append(rec_C_PSNR)

                rec_A_SSIM = compute_measure(x_A, Truth_A, 3)
                rec_F_SSIM = compute_measure(x_F, Truth_F, 3)
                rec_C_SSIM = compute_measure(x_C, Truth_C, 3)

                SSIM_A_total.append(rec_A_SSIM)
                SSIM_F_total.append(rec_F_SSIM)
                SSIM_C_total.append(rec_C_SSIM)

                if self.save_image_mode:
                    # save image
                    plt.imsave(result_dir + 'A_%d.png' % image_num, x_A, cmap='gray')
                    plt.imsave(result_dir + 'F_%d.png' % image_num, x_F, cmap='gray')
                    plt.imsave(result_dir + 'C_%d.png' % image_num, x_C, cmap='gray')

            RMSE_A_mean = np.mean(RMSE_A_total)
            RMSE_F_mean = np.mean(RMSE_F_total)
            RMSE_C_mean = np.mean(RMSE_C_total)

            PSNR_A_mean = np.mean(PSNR_A_total)
            PSNR_F_mean = np.mean(PSNR_F_total)
            PSNR_C_mean = np.mean(PSNR_C_total)

            SSIM_A_mean = np.mean(SSIM_A_total)
            SSIM_F_mean = np.mean(SSIM_F_total)
            SSIM_C_mean = np.mean(SSIM_C_total)

            output_data = "RMSE_A: %.6f | RMSE_F: %.6f | RMSE_C: %.6f | PSNR_A: %.6f | PSNR_F: %.6f | PSNR_C: %.6f | SSIM_A: %.6f | SSIM_F: %.6f | SSIM_C: %.6f " % \
                          (RMSE_A_mean, RMSE_F_mean, RMSE_C_mean, PSNR_A_mean, PSNR_F_mean, PSNR_C_mean, SSIM_A_mean, SSIM_F_mean, SSIM_C_mean)
            print(output_data)


