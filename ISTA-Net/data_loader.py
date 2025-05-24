import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader,TensorDataset
import scipy.io as sio
import h5py  #用于读写HDF5文件的库
from sklearn.model_selection import train_test_split
####################################################################################
class RandomDataset(Dataset):   #定义了一个数据集，其中每个项目都是一个随机张量
    def __init__(self, data, length):
        self.data = data
        self.len = length
    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()
    def __len__(self):
        return self.len
#####################################################################################
# loader train,val,test
def dataset_loder(state, train_A, train_F, train_C, batch_size, num_work):   #此函数为训练、验证和测试创建数据加载器，接受输入，如状态（train、val、test）、数据和批量大小
    train_A_T = torch.from_numpy(train_A)
    train_F_T = torch.from_numpy((train_F))
    train_C_T = torch.from_numpy((train_C))
    dataset = TensorDataset(train_A_T, train_F_T, train_C_T)   #将数据转换为PyTorch张量，并创建一个“TensorDataset”
    if state =='train':   #对于训练状态，会进行混洗，即打乱。对于其他状态则不进行混洗
        dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=True,num_workers=num_work)
    else:
        dataloader = DataLoader(dataset, batch_size=1, num_workers=num_work)
    return dataloader
#####################################################################################
def data_split(data):
    restore_train = data[:500, :, :]   #恢复
    print('Splited train restore', restore_train.shape)
    restore_val = data[500:600, :, :]
    print('Splited val restore', restore_val.shape)
    restore_test = data[600:700, :, :]
    print('Splited test restore', restore_test.shape)
    decompose_train = data[700:900, :, :]   #分解
    print('Splited train decompose', decompose_train.shape)
    decompose_val = data[900:1000, :, :]
    print('Splited val decompose', decompose_val.shape)
    return restore_train,restore_val,restore_test,decompose_train, decompose_val
#####################################################################################
# ct train dataloder
def CT_dataloader(num_work,batch_size):
    # Load Adipose data
    Phantom_Adipose_Name = '/20134138/AAPM2022/Phantom_Adipose.npy'
    Phantom_Adipose = np.load(Phantom_Adipose_Name)
    print('Phantom Adipose shape', np.array(Phantom_Adipose).shape)

    # load Fibroglandular_image
    Phantom_Fibroglandular_Name = '/20134138/AAPM2022/Phantom_Fibroglandular.npy'
    Phantom_Fibroglandular = np.load(Phantom_Fibroglandular_Name)
    print('Phantom_Fibroglandular shape', np.array(Phantom_Fibroglandular).shape)

    # load Calcification_image
    Phantom_Calcification_Name = '/20134138/AAPM2022/Phantom_Calcification.npy'
    Phantom_Calcification = np.load(Phantom_Calcification_Name)
    print('Phantom_Calcification shape', np.array(Phantom_Calcification).shape)

    # data split
    re_train_A, re_val_A, re_test_A, de_train_A, de_val_A = data_split(Phantom_Adipose)
    re_train_F, re_val_F, re_test_F, de_train_F, de_val_F = data_split(Phantom_Fibroglandular)
    re_train_C, re_val_C, re_test_C, de_train_C, de_val_C =  data_split(Phantom_Calcification)

    # dataloader
    train_re_d = dataset_loder('train', re_train_A,  re_train_F,  re_train_C, batch_size, num_work)
    val_re_d   = dataset_loder('val', re_val_A, re_val_F, re_val_C, batch_size, num_work)
    test_re_d  = dataset_loder('test', re_test_A,re_test_F, re_test_C, batch_size, num_work)
    train_de_d = dataset_loder('train', de_train_A, de_train_F, de_train_C, batch_size, num_work)
    val_de_d   = dataset_loder('val', de_val_A, de_val_F, de_val_C, batch_size, num_work)

    return train_re_d, val_re_d, test_re_d ,train_de_d, val_de_d
#####################################################################################
