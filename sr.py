import numpy as np
import torch
import math
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from skimage.transform import resize

class one_conv(nn.Module):
    def __init__(self,inchanels,growth_rate,kernel_size = 3):
        super(one_conv,self).__init__()
        self.conv = nn.Conv2d(inchanels,growth_rate,kernel_size=kernel_size,padding = kernel_size>>1,stride= 1)
        self.relu = nn.ReLU()
    def forward(self,x):
        output = self.relu(self.conv(x))
        return torch.cat((x,output),1)

class RDB(nn.Module):
    def __init__(self,G0,C,G,kernel_size = 3):
        super(RDB,self).__init__()
        convs = []
        for i in range(C):
            convs.append(one_conv(G0+i*G,G))
        self.conv = nn.Sequential(*convs)
        #local_feature_fusion
        self.LFF = nn.Conv2d(G0+C*G,G0,kernel_size = 1,padding = 0,stride =1)
    def forward(self,x):
        out = self.conv(x)
        lff = self.LFF(out)
        #local residual learning
        return lff + x

class rdn(nn.Module):
    def __init__(self):
        '''
        opts: the system para
        '''
        super(rdn,self).__init__()
        '''
        D: RDB number 20
        C: the number of conv layer in RDB 6
        G: the growth rate 32
        G0:local and global feature fusion layers 64filter
        '''
        self.D = 20
        self.C = 6
        self.G = 32
        self.G0 = 64
        
        kernel_size = 3
        input_channels = 1
        #shallow feature extraction 
        self.SFE1 = nn.Conv2d(input_channels,self.G0,kernel_size=kernel_size,padding = kernel_size>>1,stride=  1)
        self.SFE2 = nn.Conv2d(self.G0,self.G0,kernel_size=kernel_size,padding = kernel_size>>1,stride =1)
        #RDB for paper we have D RDB block
        self.RDBS = nn.ModuleList()
        for d in range(self.D):
            self.RDBS.append(RDB(self.G0,self.C,self.G,kernel_size))
        #Global feature fusion
        self.GFF = nn.Sequential(
               nn.Conv2d(self.D*self.G0,self.G0,kernel_size = 1,padding = 0 ,stride= 1),
               nn.Conv2d(self.G0,self.G0,kernel_size,padding = kernel_size>>1,stride = 1),
        )
        #upsample net 
        self.up_net = nn.Sequential(
                nn.Conv2d(self.G0,self.G*4,kernel_size=kernel_size,padding = kernel_size>>1,stride = 1),
                nn.PixelShuffle(2),
                nn.Conv2d(self.G,1,kernel_size=kernel_size,padding = kernel_size>>1,stride = 1)
        )
        #init
        for para in self.modules():
            if isinstance(para,nn.Conv2d):
                nn.init.orthogonal_(para.weight)
                if para.bias is not None:
                    para.bias.data.zero_()

    def forward(self,x):
        #f-1
        f__1 = self.SFE1(x)
        out  = self.SFE2(f__1)
        RDB_outs = []
        for i in range(self.D):
            out = self.RDBS[i](out)
            RDB_outs.append(out)
        out = torch.cat(RDB_outs,1)
        out = self.GFF(out)
        out = f__1+out
        out = self.up_net(out)
        return out

'''
class super_resolution():
    def __init__(self,dat,device):
        self.dat = dat
        self.device = device
        
    def run(self):
        net = rdn()
        net.load_state_dict(torch.load('./net/rdn_ct_sr_21.pkl'))
        net = net.to(self.device)

        X0 = self.dat

        with torch.no_grad():
            Xt0 = Variable(torch.from_numpy(X0)).reshape(1,1,128,128)
            Xt0 = Xt0.to(self.device)
            Xt0 = Xt0.type(torch.cuda.FloatTensor)
            Y_hat = net(Xt0)
            Y_hat = Y_hat.data.cpu().numpy()
            Y_hat = Y_hat.reshape(512,512)
            Y_hat[Y_hat<0.5] = 0
            Y_hat[Y_hat>0.5] = 1

        return Y_hat
'''

def super_resolution(data,device):
    lr_size = 256
    net = rdn()
    net.load_state_dict(torch.load('./pre-trained-weights/sr-net.pkl',map_location={'cuda:1':'cuda:0'}))
    net = net.to(device)

    with torch.no_grad():
        Xt0 = Variable(torch.from_numpy(data)).reshape((1,1,lr_size,lr_size))
        Xt0 = Xt0.to(device)
        Xt0 = Xt0.type(torch.cuda.FloatTensor)
        Y_hat = net(Xt0)
        Y_hat = Y_hat.data.cpu().numpy()
        Y_hat = Y_hat.reshape(512,512)
        #Y_hat[Y_hat<0.5] = 0
        #Y_hat[Y_hat>0.5] = 1

    return Y_hat
