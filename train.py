#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
import torch
import h5py
import os
import sys

import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch import nn, optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from datetime import datetime
from scipy import io

import parameters
from model import DeepRFT as myNet  ##等之后确定名字后更改
from funcs import calcScore,get_coeff
## current time
print('current time:',datetime.now())

## specify gpu for training
os.environ["CUDA_VISIBLE_DEVICES"]="0"
n_gpu=torch.cuda.device_count()
print('use %s GPUs for training......'%(n_gpu))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.append("..") 

## define the network
net = myNet()

## load dataset
sample_size_train = parameters.sample_size_train
sample_size_val=parameters.sample_size_val
sample_id_val=parameters.sample_id_val

## get coefficient
coeff=get_coeff(parameters.group_number)


## load training set
x = np.zeros([sample_size_train, 1,parameters.img_resolution, parameters.img_resolution])
y2 = np.zeros([sample_size_train, 1,parameters.img_resolution, parameters.img_resolution])

with h5py.File(parameters.data_path, 'r') as hf: 
    x[:,0,:,:] = hf['Bp_results'][0:sample_size_train,:,:]/coeff
    y2[:,0,:,:] = hf['groundtruth_img'][0:sample_size_train,:,:]


## load validation set
X0 = np.empty([sample_size_val,1,parameters.img_resolution,parameters.img_resolution])  
Y2 = np.empty([sample_size_val,1,parameters.img_resolution,parameters.img_resolution])

with h5py.File(parameters.val_data_path, 'r') as hf: 
    X0[:,0,:,:] = hf['Bp_results'][sample_id_val:sample_id_val+sample_size_val,:,:]/coeff
    Y2[:,0,:,:] = hf['groundtruth_img'][sample_id_val:sample_id_val+sample_size_val,:,:]
    

## define the dataset
class MyDataset(Dataset):
    
    def __init__(self, a, b):
        self.data_1 = a
        self.data_2 = b  
        #self.data_3 = c

    def __len__(self):
        return len(x)

    def __getitem__(self, idx):
        in_put = self.data_1[idx]
        out_put = self.data_2[idx]
        #out_put2 = self.data_3[idx]
        return in_put, out_put#, out_put2

## define the trainer
batchsize = parameters.batchsize
dataset = MyDataset(x,y2)
train_iter = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=10, drop_last=False, pin_memory=True)

## define the optimizer
lr=parameters.learning_rate
num_epochs=parameters.num_epochs
optimizer = torch.optim.Adam(net.parameters(), lr = lr, betas=(0.9, 0.999), eps=1e-8)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=16, eta_min=parameters.end_lr)


## check point
if parameters.checkpoint_epoch>0:
    net.load_state_dict(torch.load(parameters.result_path+str(parameters.checkpoint_epoch)+'.pkl'))

## map the net to device
print("training on ", device)
net = net.to(device)

## define loss function
loss_res = np.zeros(num_epochs) 
valida_res = np.zeros(num_epochs)

loss = torch.nn.MSELoss(reduction='sum')

## map the validation data to device
Xt0 = Variable(torch.from_numpy(X0))
Xt0 = Xt0.to(device)
Xt0 = Xt0.type(torch.cuda.FloatTensor)

## training
for epoch in range(num_epochs):
    print('training epoch:',epoch)
    train_l_sum = 0.0
    start = time.time()
    batch_count = 0
    
    for X5, Y52 in train_iter:
        ## map current batch to device
        X5 = X5.to(device).type(torch.cuda.FloatTensor)
        Y52 = Y52.to(device).type(torch.cuda.FloatTensor)
        ## forward
        y_hat52 = net(X5)
        ## loss
        l = loss(torch.squeeze(y_hat52), torch.squeeze(Y52))
        ## backward
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        
        train_l_sum += l.cpu().item()
        batch_count += 1
    
    scheduler.step()
        
    with torch.no_grad():

        Y_hat2 = net(Xt0)
        
        Y_hat2 = Y_hat2.data.cpu().numpy()
        y_hat2 = Y_hat2.reshape((sample_size_val,1,parameters.img_resolution,parameters.img_resolution))
        ## post processing
        y_hat2=y_hat2/np.max(y_hat2.flatten())
        y_hat2[y_hat2<0.5]=0
        y_hat2[y_hat2>0.5]=1
        score = np.mean(calcScore(y_hat2,Y2))
        
    print('epoch %d, loss %.6f, validation-psnr %.6f, time %.1f sec'
          % (epoch +parameters.checkpoint_epoch+ 1, train_l_sum/batch_count , score, time.time() - start))

    loss_res[epoch]=train_l_sum/batch_count 
    valida_res[epoch]=score
    if ((epoch+1) % 10) == 0:
        torch.save(net.state_dict(), parameters.result_path+str(epoch+parameters.checkpoint_epoch+1)+'.pkl')
        io.savemat(parameters.result_path+'training_epoch.mat',{'loss_res':loss_res,'valida_res':valida_res})


print('smallest error on training set:',np.argmin(loss_res)+1,'highest score on validation set:',np.argmax(valida_res)+1)
print('current time:',datetime.now())
