import torch
import torch.nn as nn
import numpy as np
from constants import *


class EMGNet(nn.Module):
    def __init__(self, d_e, train=True, device="cuda"):
        super(EMGNet,self).__init__()
        self.device=device
        self.d_e=d_e

        # momentum = 0 and batch per subject in order to have adaptive normalization (https://doi.org/10.1016/j.patcog.2018.03.005)

        # similar architecture to https://doi.org/10.3390/s17030458
        self.conv_emg=nn.Sequential(
                # conv -> bn -> relu
                nn.BatchNorm2d(1,momentum=0), # normalize input based on batch

                nn.Conv2d(1,64,(3,1),padding=(1,0)),     # maintain same shape
                nn.BatchNorm2d(64,momentum=0),
                nn.LeakyReLU(),

                nn.Conv2d(64,64,(3,1),padding=(1,0)),
                nn.BatchNorm2d(64,momentum=0),
                nn.LeakyReLU(),

                nn.Conv2d(64,64,(1,1)),
                nn.BatchNorm2d(64,momentum=0),
                nn.ReLU(),
                nn.Dropout(p=.1),

                nn.Conv2d(64,64,(1,1)),
                nn.BatchNorm2d(64,momentum=0),
                nn.ReLU(),
                nn.Dropout(p=.1),

                nn.Flatten(),

                nn.Linear(WINDOW_MS*EMG_DIM*64, 512),
                nn.BatchNorm1d(512,momentum=0),
                nn.ReLU(),
                nn.Dropout(p=.5),
                )

        self.feedforward_acc=nn.Sequential(
                nn.BatchNorm1d(ACC_DIM,momentum=0),
                nn.Linear(ACC_DIM, 256),
                nn.BatchNorm1d(256,momentum=0),
                nn.LeakyReLU(),

                nn.Linear(256, 128),
                nn.BatchNorm1d(128,momentum=0),
                nn.ReLU(),
                nn.Dropout(),

                nn.Linear(128, 128),
                nn.BatchNorm1d(128,momentum=0),
                nn.ReLU(),
                )

        self.fusion_head = nn.Sequential(
                nn.Linear(512+128, 512),
                nn.BatchNorm1d(512,momentum=0),
                nn.ReLU(),
                nn.Dropout(p=.5),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256,momentum=0),
                nn.ReLU(),

                )

        self.proj_mat = nn.Linear(256, self.d_e)

    def forward(self, EMG, ACC):
        ACC=ACC.squeeze(1)
        out_emg=self.conv_emg(EMG)
        out_acc=self.feedforward_acc(ACC)
        out=torch.cat((out_emg,out_acc),dim=1)
        out=self.fusion_head(out)
        out=self.proj_mat(out)
        return out

    def l2(self):
        reg_loss = 0
        for name,param in self.named_parameters():
            if 'bn' not in name:
                reg_loss+=torch.norm(param)
        return reg_loss
        


class GLOVENet(nn.Module):
    def __init__(self, d_e, train=True, device="cuda"):
        super(GLOVENet,self).__init__()
        self.device=device
        self.d_e=d_e

        # momentum = 0 and batch per subject in order to have adaptive normalization (https://doi.org/10.1016/j.patcog.2018.03.005)

        self.conv=nn.Sequential(
                # conv -> bn -> relu
                nn.BatchNorm2d(1,momentum=0), # normalize input based on batch

                nn.Conv2d(1,64,(3,1),padding=(1,0)),     # maintain same shape
                nn.BatchNorm2d(64,momentum=0),
                nn.LeakyReLU(),

                nn.Conv2d(64,32,(1,1)),
                nn.BatchNorm2d(32,momentum=0),
                nn.LeakyReLU(),
                nn.Dropout(p=.5),

                nn.Flatten(),

                nn.Linear(WINDOW_MS*GLOVE_DIM*32, 256),
                nn.BatchNorm1d(256,momentum=0),
                nn.ReLU(),
                )

        self.proj_mat = nn.Linear(256, self.d_e)

    def forward(self, GLOVE):
        out=self.conv(GLOVE)
        out=self.proj_mat(out)
        return out

    def l2(self):
        reg_loss = 0
        for name,param in self.named_parameters():
            if 'bn' not in name:
                reg_loss+=torch.norm(param)
        return reg_loss


