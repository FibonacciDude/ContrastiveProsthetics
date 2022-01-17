import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from constants import *
from utils import RunningStats

#torch.autograd.set_detect_anomaly(True)

# adaptive batch normalization - https://doi.org/10.1016/j.patcog.2018.03.005
class AdaBatchNorm1d(nn.Module):
    # No code for the online mean/std at test time yet...
    def __init__(self, num_features, device="cuda"):
        super(AdaBatchNorm1d, self).__init__()
        self.device=torch.device(device)
        self.bn = nn.BatchNorm1d(num_features=num_features, momentum=0, track_running_stats=False)
        self.to(self.device)
    def forward(self, X):
        return self.bn(X)
        
class AdaBatchNorm2d(nn.Module):
    # No code for the online mean/std at test time yet...
    def __init__(self, num_features, device="cuda"):
        super(AdaBatchNorm2d, self).__init__()
        self.device=torch.device(device)
        self.bn = nn.BatchNorm2d(num_features=num_features, momentum=0, track_running_stats=False)
        self.to(self.device)
    def forward(self, X):
        return self.bn(X)

class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()
    
    def forward(self, X):
        print(X.shape)
        return X

class LocalLinear(nn.Module):
    def __init__(self,in_features,local_features,kernel_size,padding=0,stride=1,bias=True):
        super(LocalLinear, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        fold_num = (in_features+2*padding-self.kernel_size)//self.stride+1
        self.weight = nn.Parameter(torch.randn(fold_num,kernel_size,local_features))
        self.bias = nn.Parameter(torch.randn(fold_num,local_features)) if bias else None

    def forward(self, x:torch.Tensor):
        x = F.pad(x,[self.padding]*2,value=0)
        x = x.unfold(-1,size=self.kernel_size,step=self.stride)
        x = torch.matmul(x.unsqueeze(2),self.weight).squeeze(2)
        if self.bias is not None:
            x = x + self.bias
        return x


# modeled after https://github.com/openai/CLIP/blob/main/clip/model.py
class Model(nn.Module):
    def __init__(self, d_e, dp=.5, adabn=True, train_model=True, device="cuda"):
        super(Model,self).__init__()

        self.train_model = train_model
        self.d_e = d_e
        self.adabn=adabn
        self.device = torch.device(device)

        self.emg_net = EMGNet(d_e=d_e, dp=dp, adabn=adabn) # emg model

        self.loss_f = torch.nn.functional.cross_entropy
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1)/0.07)    # CLIP logit scale
        self.to(self.device)

        self.correct_tr = []
        self.correct_v = []

    def set_train(self):
        self.train_model=True
        self.train()

    def set_test(self):
        self.train_model=False
        self.eval()

    def set_val(self):
        self.set_test() # same behavior

    def encode_emg(self, EMG):
        return self.emg_net(EMG)

    def forward(self, EMG):
        emg_features = self.encode_emg(EMG)
        emg_features = emg_features / emg_features.norm(dim=-1,keepdim=True)
        return emg_features

    def loss(self, features, labels):
        loss = self.loss_f(features, labels)
        correct = (F.softmax(features, dim=-1).detach().cpu().numpy().argmax(-1)==labels.cpu().numpy()).mean()
        if self.train_model:
            self.correct_v = []
            self.correct_tr.append(correct.item())
        else:
            self.correct_tr = []
            self.correct_v.append(correct.item())
        return loss

    def correct_glove(self):
        if self.train_model:
            return np.array(self.correct_tr).mean()
        return np.array(self.correct_v).mean()

    def l2(self):
        return self.emg_net.l2()

class EMGNet(nn.Module):
    def __init__(self, d_e, dp=.5, adabn=True, train=True, device="cuda"):
        super(EMGNet,self).__init__()
        self.device=torch.device(device)
        self.d_e=d_e
        self.dp=dp
        if adabn:
            self.bn1d_func = AdaBatchNorm1d
            self.bn2d_func = AdaBatchNorm2d
        else:
            self.bn1d_func = nn.BatchNorm1d
            self.bn2d_func = nn.BatchNorm2d

        # momentum = 0 and batch per subject in order to have adaptive normalization (https://doi.org/10.1016/j.patcog.2018.03.005)
        # loosely inspired by architecture in https://doi.org/10.3390/s17030458

        self.conv_emg=nn.Sequential(
                # conv -> bn -> relu
                #self.bn2d_func(1),

                # prevent premature fusion (https://www.mdpi.com/2071-1050/10/6/1865/htm) 
                # larger kernel
                nn.Conv2d(1,64,(3,3),padding=(1,1)),
                nn.ReLU(),
                self.bn2d_func(64),

                nn.Conv2d(64,64,(3,3),padding=(1,1)),
                nn.ReLU(),
                self.bn2d_func(64),
                nn.Flatten(),
                )

        self.linear = nn.Sequential(
                nn.Linear(EMG_DIM*64, 512),
                nn.ReLU(),
                self.bn1d_func(512),

                nn.Linear(512, 512),
                nn.ReLU(),
                self.bn1d_func(512),

                nn.Linear(512, 512),
                nn.ReLU(),
                self.bn1d_func(512),
                nn.Dropout(self.dp),

                nn.Linear(512, 512),
                nn.ReLU(), 
                self.bn1d_func(512),
                nn.Dropout(self.dp),

                nn.Linear(512, 128),
                nn.ReLU(),
                self.bn1d_func(128),
                nn.Dropout(self.dp),

                nn.Linear(128, 37),
                )

        self.to(self.device)

    def forward(self, EMG):
        out=self.conv_emg(EMG)
        out=self.linear(out)
        return out

    def l2(self):
        reg_loss = 0
        for name,param in self.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                reg_loss+=torch.norm(param)
        return reg_loss
