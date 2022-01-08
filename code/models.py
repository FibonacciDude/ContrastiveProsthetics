import torch
import torch.nn as nn
import numpy as np
from constants import *
from utils import RunningStats

class AdaBN1d(nn.Module):
    def __init__(self, device="cuda"):
        super(EMGNet,self).__init__()
        self.device=torch.device(device)
        self.buffer = []
        self.subjects = []
        self.to(device)

    def forward(self, X, subject):
        if subject not in self.subjects:
            self.subjects.add(subject)
            self.buffer.add(RunningStats(X))


class EMGNet(nn.Module):
    def __init__(self, d_e, train=True, device="cuda"):
        super(EMGNet,self).__init__()
        self.device=torch.device(device)
        self.d_e=d_e

        # momentum = 0 and batch per subject in order to have adaptive normalization (https://doi.org/10.1016/j.patcog.2018.03.005)

        # similar architecture to https://doi.org/10.3390/s17030458

        self.conv_emg=nn.Sequential(
                # conv -> bn -> relu
                nn.BatchNorm2d(1,momentum=0,track_running_stats=False),

                # prevent premature fusion (https://www.mdpi.com/2071-1050/10/6/1865/htm) 
                # larger kernel
                # TODO: add muscle independence
                nn.Conv2d(1,64,(5,3),padding=(2,1)),
                nn.BatchNorm2d(64,momentum=0,track_running_stats=False),
                nn.LeakyReLU(),

                # smaller kernel
                nn.Conv2d(64,64,(3,3),padding=(1,1)),
                nn.BatchNorm2d(64,momentum=0,track_running_stats=False),
                nn.LeakyReLU(),

                nn.Conv2d(64,64,(1,1)),
                nn.BatchNorm2d(64,momentum=0,track_running_stats=False),
                nn.LeakyReLU(),

                # one more layer (why not?, it acts like a bottleneck)
                nn.Conv2d(64,64,(3,3),padding=(1,1)),
                nn.BatchNorm2d(64,momentum=0,track_running_stats=False),
                nn.LeakyReLU(),
                nn.Dropout(p=.5),

                nn.Conv2d(64,64,(1,1)),
                nn.BatchNorm2d(64,momentum=0,track_running_stats=False),
                nn.LeakyReLU(),
                # WINDOW_MS x EMG_DIM -> 1 x EMG_DIM
                nn.AvgPool2d((WINDOW_MS, 1)),
                #nn.AvgPool2d((WINDOW_MS//5, 1), stride=(2, 1)),
                nn.Dropout(p=.5),

                nn.Flatten(),

                nn.Linear(EMG_DIM*64, 512),
                #nn.Linear(EMG_DIM*64*((WINDOW_MS-WINDOW_MS//5)//2+1), 512),
                nn.BatchNorm1d(512, momentum=0,track_running_stats=False),
                nn.LeakyReLU(),
                )

        # No fusion for now. 
        # Reason: The classification might be affected with different activites
        # and the acceleration data shifts for different windows (relevant to static)
        self.use_acc = False

        if self.use_acc:
            self.feedforward_acc=nn.Sequential(
                    nn.BatchNorm1d(ACC_DIM,momentum=0),
                    nn.Linear(ACC_DIM, 256),
                    nn.BatchNorm1d(256,momentum=0,track_running_stats=False),
                    nn.LeakyReLU(),

                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128,momentum=0,track_running_stats=False),
                    nn.LeakyReLU(),
                    nn.Dropout(),

                    nn.Linear(128, 128),
                    nn.BatchNorm1d(128,momentum=0,track_running_stats=False),
                    nn.LeakyReLU(),
                    )

            self.fusion_head = nn.Sequential(
                    nn.BatchNorm1d(1024+128,momentum=0,track_running_stats=False),
                    nn.LeakyReLU(),
                    nn.Linear(1024+128, 512),
                    nn.Dropout(p=.5),

                    nn.BatchNorm1d(512,momentum=0,track_running_stats=False),
                    nn.LeakyReLU(),
                    )

        self.proj_mat = nn.Linear(512, self.d_e)

    def forward(self, EMG, ACC):
        ACC=ACC.squeeze(1)
        out_emg=self.conv_emg(EMG)
        if self.use_acc:
            out_acc=self.feedforward_acc(ACC)
            out=torch.cat((out_emg,out_acc),dim=1)
            out=self.fusion_head(out)
        else:
            out=out_emg
        out=self.proj_mat(out)
        return out

    def l2(self):
        reg_loss = 0
        for name,param in self.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                reg_loss+=torch.norm(param)
        return reg_loss
        


class GLOVENet(nn.Module):
    def __init__(self, d_e, train=True, device="cuda"):
        super(GLOVENet,self).__init__()
        self.device=device
        self.d_e=d_e

        # momentum = 0 and batch per subject in order to have adaptive normalization (https://doi.org/10.1016/j.patcog.2018.03.005)

        # This network is someone large compared to the complexity of the input data
        # in order to prevent underfitting. This need to have the same representation 
        # as the EMG conv net layer so it must be similarly powerful.
        # TODO: Make smaller though? (I think it might be overfitting)

        # There should not be any temporal information as the Hz of the glove is 
        # much lower. Take the mean.

        GLOVE=GLOVE.mean(dim=2, keepdim=True)

        self.conv=nn.Sequential(
                # Initial glove features
                nn.Conv2d(1,512,(1, GLOVE_DIM),padding=(0,0)),
                # 3 (in the 15 ms case) x 1 
                #nn.BatchNorm2d(512,momentum=.1),
                nn.BatchNorm2d(512, momentum=0, track_running_stats=False),
                nn.LeakyReLU(),
                nn.Flatten(),

                nn.Linear(512, 512),
                nn.BatchNorm1d(512, momentum=0, track_running_stats=False),
                nn.LeakyReLU(),
                nn.Dropout(p=.5),
                
                nn.Linear(512, 512),
                nn.BatchNorm1d(512, momentum=0, track_running_stats=False),
                nn.LeakyReLU(),
                nn.Dropout(p=.5),

                nn.Linear(512, 512),
                nn.BatchNorm1d(512, momentum=0, track_running_stats=False),
                nn.LeakyReLU(),
                )

        self.proj_mat = nn.Linear(512, self.d_e)

    def forward(self, GLOVE):
        out=self.conv(GLOVE)
        out=self.proj_mat(out)
        return out

    def l2(self):
        reg_loss = 0
        for name,param in self.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                reg_loss+=torch.norm(param)
        return reg_loss

