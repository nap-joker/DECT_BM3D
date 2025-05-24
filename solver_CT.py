import os
import torch
import torch.nn as nn
from functions import * #自定义的函数模块
import glob   #用于从文件夹中读取文件列表
from skimage.measure import compare_ssim as ssim
from functions import *
import cv2
from time import time
import matplotlib.pyplot as plt
import random
from numpy import *
import numpy as np
import odl  #Optimal design library 用于数学建模和优化问题
import odl.contrib.torch as odl_torch   #ODL库和PyTorch的集成
###########################################################################
def dualEnergyTransmission(sa, sf, sc, delta_e, modeldata,device):   #这个函数计算了双能CT图像的传输，接受一些参数，包括射线数据（sa、sf、sc）、能量差值和模型数据
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
        # # 防止exp
        exp_tmp[exp_tmp>5] = 5  # 将数组里面的无穷值转为空值
        # trans_tmp = torch.mul(delta_e, spectrum[j])
        # transmission = transmission + torch.mul(trans_tmp, exp(exp_tmp))
        transmission += delta_e * spectrum[j] * torch.exp(exp_tmp)
    return transmission   #根据输入的参数计算传输，并返回传输张量
###########################################################################
def projection_operator(ximageside,yimageside,Truth_A_T,slen,nviews,nbins,radius,source_to_detector):   #用于创建投影算子，将原始图像转换为投影数据
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
class Solver_CT(object):   #CT图像重建任务的求解器
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
        self.layer_num = args.layer_num
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
        self.modeldata_low = load("./training_data/model_data_" + self.kvp_low + "kVp.npy")
        self.modeldata_high = load("./training_data/model_data_" + self.kvp_high + "kVp.npy")

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
        # define save dir
        model_dir = "./%s/%s_layer_%d_lr_%f" % (
            self.model_dir, self.net_name, self.layer_num, self.lr)
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
            # model = self.model.train()
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
                    loss_discrepancy = torch.mean(torch.abs(x_output - batch_x))  # Compute and print loss
                    loss_discrepancy_mid = torch.mean(torch.abs(x_mid - batch_x))
                    loss_all = loss_discrepancy + loss_discrepancy_mid

                elif self.loss_mode == 'L1':  # simple L1
                    loss_discrepancy = torch.mean(torch.abs(x_output - batch_x))  # Compute and print loss
                    loss_all = loss_discrepancy   # Compute and print loss

                elif self.loss_mode == 'L2':  # simple L1
                    train_loss = nn.MSELoss()
                    loss_discrepancy = train_loss(x_output, batch_x)  # Compute and print loss
                    loss_all = loss_discrepancy   # Compute and print loss

                elif self.loss_mode == 'Fista-net':  # simple L1
                    # Compute loss, data consistency and regularizer constraints
                    train_loss = nn.MSELoss()
                    loss_discrepancy = train_loss(x_output, batch_x)  # + l1_loss(pred, y_target, 0.3)
                    loss_constraint = 0
                    for k, _ in enumerate(x_mid, 0):
                        loss_constraint += torch.mean(torch.pow(x_mid[k], 2))

                    encoder_constraint = 0
                    for k, _ in enumerate(x_last, 0):
                        encoder_constraint += torch.mean(torch.abs(x_last[k]))

                    loss_all = loss_discrepancy + 0.01 * loss_constraint + 0.001 * encoder_constraint

                elif self.loss_mode == 'Ista-net':  # simple L1
                    # Compute and print loss
                    loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))

                    loss_constraint = torch.mean(torch.pow(x_mid[0], 2))
                    for k in range(self.layer_num - 1):
                        loss_constraint += torch.mean(torch.pow(x_mid[k + 1], 2))

                    gamma = torch.Tensor([0.01]).to(self.device)

                    # loss_all = loss_discrepancy
                    loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint)

                # Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()
                loss_all.backward()
                self.optimizer.step()

                # step %100==0
                if step % 10 == 0:
                    output_data = "[%02d/%02d] Step:%.0f | Total Loss: %.6f | Discrepancy Loss: %.6f " % \
                                  (epoch_i, self.end_epoch, step, loss_all.item(), loss_discrepancy.item())
                    print(output_data)

            # val
            model = self.model.eval()
            # Load pre-trained model with epoch number
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
                    # 计算结果指标
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
                    rec_A_PSNR = psnr(x_A * 255.0, Truth_A * 255.0)
                    rec_F_PSNR = psnr(x_F * 255.0, Truth_F * 255.0)
                    rec_C_PSNR = psnr(x_C * 255.0, Truth_C * 255.0)

                    rec_A_SSIM = ssim(x_A, Truth_A, data_range=1)
                    rec_F_SSIM = ssim(x_F, Truth_F, data_range=1)
                    rec_C_SSIM = ssim(x_C, Truth_C, data_range=1)

                    RMSE_A_total.append(rec_A_RMSE)
                    RMSE_F_total.append(rec_F_RMSE)
                    RMSE_C_total.append(rec_C_RMSE)
                    PSNR_A_total.append(rec_A_PSNR)
                    PSNR_F_total.append(rec_F_PSNR)
                    PSNR_C_total.append(rec_C_PSNR)
                    SSIM_A_total.append(rec_A_SSIM)
                    SSIM_F_total.append(rec_F_SSIM)
                    SSIM_C_total.append(rec_C_SSIM)

                RMSE_A_total_mean = np.array(RMSE_A_total).mean()
                RMSE_F_total_mean = np.array(RMSE_F_total).mean()
                RMSE_C_total_mean = np.array(RMSE_C_total).mean()
                PSNR_A_total_mean = np.array(PSNR_A_total).mean()
                PSNR_F_total_mean = np.array(PSNR_F_total).mean()
                PSNR_C_total_mean = np.array(PSNR_C_total).mean()
                SSIM_A_total_mean = np.array(SSIM_A_total).mean()
                SSIM_F_total_mean = np.array(SSIM_F_total).mean()
                SSIM_C_total_mean = np.array(SSIM_C_total).mean()

                # 打印中间结果
            print("RMSE_A is %.5f, RMSE_F is %.5f, RMSE_C is %.5f, PSNR_A is %.2f, PSNR_F is %.2f,"
                  "PSNR_C is %.2f, SSIM_A is %.4f, SSIM_F is %.4f, SSIM_A is %.4f" %
                  (RMSE_A_total_mean, RMSE_F_total_mean, RMSE_C_total_mean, PSNR_A_total_mean, PSNR_F_total_mean,
                   PSNR_C_total_mean, SSIM_A_total_mean, SSIM_F_total_mean, SSIM_C_total_mean))

            # save model in every epoch
            RMSE_mean = (RMSE_A_total_mean + RMSE_F_total_mean + RMSE_C_total_mean) / 2
            if RMSE_mean < best_RMSE:
                print('RMSE_mean:{} < best_RMSE:{}'.format(RMSE_mean, best_RMSE))
                best_RMSE = RMSE_mean
                print('===========>save best model!')
                torch.save(self.model.state_dict(),
                           "./%s/net_params_%d.pkl" % (model_dir, self.end_epoch))  # save only the parameters

            # save result
            output_data = [epoch_i, loss_all.item(), RMSE_A_total_mean, RMSE_F_total_mean, RMSE_C_total_mean,
                           PSNR_A_total_mean, PSNR_F_total_mean,
                           PSNR_C_total_mean, SSIM_A_total_mean, SSIM_F_total_mean, SSIM_C_total_mean]
            output_file_name = "./%s/%s_lr_%f.txt" % (
                self.log_dir, self.net_name, self.lr)
            output_file = open(output_file_name, 'a')
            for fp in output_data:  # write data in txt
                output_file.write(str(fp))
                output_file.write(',')
            output_file.write('\n')  # line feed
            output_file.close()

    def test(self):
        # Load pre-trained model with epoch number
        model_dir = "%s/%s_layer_%d_lr_%f" % (
            self.model_dir, self.net_name, self.layer_num, self.lr)
        self.model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, self.end_epoch)))
        # Load pre-trained model with epoch number
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
            time_all = 0
            for i, data in enumerate(self.test_loader):
                image_num = image_num + 1
                Truth_A_T, Truth_F_T, Truth_C_T = data
                Truth_A_T = torch.unsqueeze(Truth_A_T, 1).to(self.device)
                Truth_F_T = torch.unsqueeze(Truth_F_T, 1).to(self.device)
                Truth_C_T = torch.unsqueeze(Truth_C_T, 1).to(self.device)

                # creat low/high FBP
                primal_op_layer, dual_op_layer = \
                    projection_operator(self.ximageside, self.yimageside, Truth_A_T, self.slen, self.nviews,
                                        self.nbins, self.radius, self.source_to_detector)

                # predict
                start = time()
                [x_output, x_mid, x_last] = model(Truth_A_T, Truth_F_T, Truth_C_T, primal_op_layer,
                                               dual_op_layer, dualEnergyTransmission, self.model_D,
                                               self.device, self.delta_e, self.modeldata_low,
                                               self.modeldata_high)
                end = time()
                time_all = time_all + end - start
                # 计算结果指标
                Truth_A = torch.squeeze(Truth_A_T).cpu().data.numpy()
                Truth_F = torch.squeeze(Truth_F_T).cpu().data.numpy()
                Truth_C = torch.squeeze(Truth_C_T).cpu().data.numpy()

                x_A = x_output[:, 0:1, :, :]
                x_F = x_output[:, 1:2, :, :]
                x_C = x_output[:, 2:3, :, :]
                x_A = torch.squeeze(x_A).cpu().data.numpy()
                x_F = torch.squeeze(x_F).cpu().data.numpy()
                x_C = torch.squeeze(x_C).cpu().data.numpy()

                rec_A_RMSE = compute_measure(x_A, Truth_A, 1)
                rec_F_RMSE = compute_measure(x_F, Truth_F, 1)
                rec_C_RMSE = compute_measure(x_C, Truth_C, 1)
                rec_A_PSNR = psnr(x_A * 255.0, Truth_A * 255.0)
                rec_F_PSNR = psnr(x_F * 255.0, Truth_F * 255.0)
                rec_C_PSNR = psnr(x_C * 255.0, Truth_C * 255.0)
                rec_A_SSIM = ssim(x_A, Truth_A, data_range=1)
                rec_F_SSIM = ssim(x_F, Truth_F, data_range=1)
                rec_C_SSIM = ssim(x_C, Truth_C, data_range=1)

                RMSE_A_total.append(rec_A_RMSE)
                RMSE_F_total.append(rec_F_RMSE)
                RMSE_C_total.append(rec_C_RMSE)
                PSNR_A_total.append(rec_A_PSNR)
                PSNR_F_total.append(rec_F_PSNR)
                PSNR_C_total.append(rec_C_PSNR)
                SSIM_A_total.append(rec_A_SSIM)
                SSIM_F_total.append(rec_F_SSIM)
                SSIM_C_total.append(rec_C_SSIM)

                # save image
                if self.save_image_mode == 1:
                    img_dir_A = result_dir + str(image_num) + "_A_PSNR_%.3f_SSIM_%.5f_RMSE_%.5f.png" % (
                        rec_A_PSNR, rec_A_SSIM, rec_A_RMSE)
                    print(img_dir_A)
                    im_rec_rgb_A = np.clip(x_A * 255, 0, 255).astype(np.uint8)
                    cv2.imwrite(img_dir_A, im_rec_rgb_A)
                    print( "Mean run time for %s test is %.6f, Proposed PSNR is %.3f, SSIM is %.5f, Proposed RMSE is %.5f" % (
                            image_num, time_all/image_num , rec_A_PSNR, rec_A_SSIM, rec_A_RMSE))

                    img_dir_F = result_dir + str(image_num) + "_F_PSNR_%.3f_SSIM_%.5f_RMSE_%.5f.png" % (
                        rec_F_PSNR, rec_F_SSIM, rec_F_RMSE)
                    print(img_dir_F)
                    im_rec_rgb_F = np.clip(x_F * 255, 0, 255).astype(np.uint8)
                    cv2.imwrite(img_dir_F, im_rec_rgb_F)
                    print(
                        "Mean run time for %s test is %.6f, Proposed PSNR is %.3f, SSIM is %.5f, Proposed RMSE is %.5f" % (
                            image_num, time_all/image_num, rec_F_PSNR, rec_F_SSIM, rec_F_RMSE))

                    img_dir_C = result_dir + str(image_num) + "_C_PSNR_%.3f_SSIM_%.5f_RMSE_%.5f.png" % (
                        rec_C_PSNR, rec_C_SSIM, rec_C_RMSE)
                    print(img_dir_C)
                    im_rec_rgb_C = np.clip(x_C * 255, 0, 255).astype(np.uint8)
                    cv2.imwrite(img_dir_C, im_rec_rgb_C)
                    print(
                        "Mean run time for %s test is %.6f, Proposed PSNR is %.3f, SSIM is %.5f, Proposed RMSE is %.5f" % (
                            image_num, time_all/image_num, rec_C_PSNR, rec_C_SSIM, rec_C_RMSE))


            RMSE_A_total_mean = np.array(RMSE_A_total).mean()
            RMSE_F_total_mean = np.array(RMSE_F_total).mean()
            RMSE_C_total_mean = np.array(RMSE_C_total).mean()
            PSNR_A_total_mean = np.array(PSNR_A_total).mean()
            PSNR_F_total_mean = np.array(PSNR_F_total).mean()
            PSNR_C_total_mean = np.array(PSNR_C_total).mean()
            SSIM_A_total_mean = np.array(SSIM_A_total).mean()
            SSIM_F_total_mean = np.array(SSIM_F_total).mean()
            SSIM_C_total_mean = np.array(SSIM_C_total).mean()

            # 打印中间结果
        print("RMSE_A is %.5f, RMSE_F is %.5f, RMSE_C is %.5f, PSNR_A is %.2f, PSNR_F is %.2f,"
              "PSNR_C is %.2f, SSIM_A is %.4f, SSIM_F is %.4f, SSIM_A is %.4f" %
              (RMSE_A_total_mean, RMSE_F_total_mean, RMSE_C_total_mean, PSNR_A_total_mean, PSNR_F_total_mean,
               PSNR_C_total_mean, SSIM_A_total_mean, SSIM_F_total_mean, SSIM_C_total_mean))

        # save result
        output_data = [ RMSE_A_total_mean, RMSE_F_total_mean, RMSE_C_total_mean,
                       PSNR_A_total_mean, PSNR_F_total_mean,
                       PSNR_C_total_mean, SSIM_A_total_mean, SSIM_F_total_mean, SSIM_C_total_mean]
        # output_file_name = "./%s/%s_lr_%f.txt" % (
        #     self.log_dir, self.net_name, self.lr)
        output_file_name = result_dir_tmp + '_' + self.net_name + '_ds_' + str(self.ds_factor) + '_epoch_' + str(
            self.end_epoch) + ".txt"
        output_file = open(output_file_name, 'a')
        for fp in output_data:  # write data in txt
            output_file.write(str(fp))
            output_file.write(',')
        output_file.write('\n')  # line feed
        output_file.close()


