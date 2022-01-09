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

# modeled after https://github.com/openai/CLIP/blob/main/clip/model.py
class Model(nn.Module):
    def __init__(self, d_e, train_model=True, device="cuda"):
        super(Model,self).__init__()

        self.train_model = train_model
        self.d_e = d_e
        self.device = torch.device(device)
        self.emg_net = EMGNet(d_e=d_e) # emg model
        self.glove_net = GLOVENet(d_e=d_e) # glove model
        self.loss_f = torch.nn.functional.cross_entropy
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1))#/0.07)    # CLIP logit scale
        self.to(self.device)

    def set_train(self):
        self.train_model=True

    def set_test(self):
        self.train_model=False

    def set_val(self):
        self.set_test() # same behavior

    def encode_emg(self, EMG, ACC):
        emg_features = self.emg_net(EMG, ACC)
        return emg_features

    def encode_glove(self, GLOVE):
        glove_features = self.glove_net(GLOVE)
        return glove_features

    def forward(self, EMG, ACC, GLOVE, EMG_T, GLOVE_T):

        emg_features = self.encode_emg(EMG, ACC)
        glove_features = self.encode_glove(GLOVE)
        emg_features = emg_features.reshape((EMG_T,-1,self.d_e)).permute((1,0,2))
        glove_features = glove_features.reshape((GLOVE_T,-1,self.d_e)).permute((1,0,2))
        #emg_features = emg_features.reshape((-1,EMG_T,self.d_e))
        #glove_features = glove_features.reshape((-1,GLOVE_T,self.d_e))
        
        emg_features = emg_features / emg_features.norm(dim=-1,keepdim=True)
        glove_features = glove_features / glove_features.norm(dim=-1,keepdim=True)

        #                        -> N_e x N_g
        # encoders give input as (TASKS*WINDOW_BLOCK, d_e) or another N
        # we want (-1, TASKS, d_e) and take cross entropy across entire (-1) dim

        logit_scale=self.logit_scale.exp().clamp(min=1e-8,max=100)
        logits = torch.matmul(emg_features, glove_features.permute(0,2,1)) # shape = (N,TASKS_e,TASKS_g)


        if self.train_model:
            return self.loss(logits * logit_scale), logits * logit_scale

        return logits

    def glove_probs(self, logits):
        # (N,tasks_e,tasks_g)
        return (100.0 * logits).softmax(dim=-1)

    def emg_probs(self, logits):
        logits = logits.permute((0,2,1))
        # (N,tasks_g,tasks_e)
        return (100.0 * logits).softmax(dim=-1)

    def predict_glove_from_emg(self, logits):
        # glove_probs gives (N,tasks_e,tasks_g)
        return self.glove_probs(logits).argmax(dim=2) # (N,tasks_e), glove pred from each emg

    def predict_emg_from_glove(self, logits):
        return self.emg_probs(logits).argmax(dim=2) # (N,tasks_g), emg pred for each glove

    def correct_glove(self, logits):
        N,tasks,tasks=logits.shape
        argmax_glove=self.predict_glove_from_emg(logits)
        labels_=self.get_labels(N,tasks)
        correct = (argmax_glove.reshape(-1)==labels_).sum().cpu().item()
        return correct

    def correct_emg(self, logits):
        N,tasks,tasks=logits.shape
        argmax_emg=self.predict_emg_from_glove(logits)
        labels_=self.get_labels(N,tasks)
        correct = (argmax_emg.reshape(-1)==labels_).sum().cpu().item()

    def get_labels(self, N, tasks):
        return torch.stack([torch.arange(tasks,dtype=torch.long,device=self.device)]*N).reshape(N*tasks)

    def loss(self, logits):

        # matrix should be symmetric
        N,tasks,tasks=logits.shape  # e x g
        labels = self.get_labels(N,tasks)
        # convert (N_e, N_g) -> (n,task_e,N_g) -> (n,task_e,n,task_g) -> (n,n,task_g,task_e) -> (n^2,task_g,task_e)
        logits_e = logits.reshape((N*tasks,tasks))
        logits_g = logits.permute((0,2,1)).reshape((N*tasks,tasks))
        loss_e = self.loss_f(logits_e, labels,reduction='mean')
        loss_g = self.loss_f(logits_g, labels,reduction='mean')
        loss = (loss_e+loss_g)/2
        return loss

    def l2(self):
        return self.emg_net.l2() + self.glove_net.l2()

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
                nn.Conv2d(1,64,(7,3),padding=(3,1), bias=False),
                nn.BatchNorm2d(64,momentum=0,track_running_stats=False),
                nn.LeakyReLU(),

                # smaller kernel
                nn.Conv2d(64,64,(3,3),padding=(1,1), bias=False),
                nn.BatchNorm2d(64,momentum=0,track_running_stats=False),
                nn.LeakyReLU(),

                nn.Conv2d(64,64,(1,1), bias=False),
                nn.BatchNorm2d(64,momentum=0,track_running_stats=False),
                nn.LeakyReLU(),
                nn.Dropout(p=.5),

                # one more layer (why not?, it acts like a bottleneck)
                nn.Conv2d(64,64,(3,3),padding=(1,1)),
                nn.BatchNorm2d(64,momentum=0,track_running_stats=False),
                nn.LeakyReLU(),
                nn.Dropout(p=.5),

                nn.Conv2d(64,64,(1,1), bias=False),
                nn.BatchNorm2d(64,momentum=0,track_running_stats=False),
                nn.LeakyReLU(),
                # WINDOW_MS x EMG_DIM -> 1 x EMG_DIM
                nn.Flatten(),

                nn.Linear(EMG_DIM*WINDOW_MS*64, 512, bias=False),
                #nn.Linear(EMG_DIM*64*((WINDOW_MS-WINDOW_MS//5)//2+1), 512),
                nn.BatchNorm1d(512, momentum=0,track_running_stats=False),
                nn.LeakyReLU(),
                )

        # No fusion for now. 
        # Reason: The classification might be affected with different activites
        # and the acceleration data shifts for different windows (relevant to static)
        self.use_acc = True # False

        if self.use_acc:
            self.feedforward_acc=nn.Sequential(
                    nn.BatchNorm1d(ACC_DIM,momentum=0),
                    nn.Linear(ACC_DIM, 256, bias=False),
                    nn.BatchNorm1d(256, momentum=0,track_running_stats=False),
                    nn.LeakyReLU(),

                    nn.Linear(256, 128, bias=False),
                    nn.BatchNorm1d(128,momentum=0,track_running_stats=False),
                    nn.LeakyReLU(),
                    nn.Dropout(),

                    nn.Linear(128, 128, bias=False),
                    nn.BatchNorm1d(128,momentum=0,track_running_stats=False),
                    nn.LeakyReLU(),
                    )

            self.fusion_head = nn.Sequential(
                    nn.BatchNorm1d(512+128,momentum=0,track_running_stats=False),
                    nn.LeakyReLU(),
                    nn.Linear(512+128, 512, bias=False),
                    nn.Dropout(p=.5),

                    nn.BatchNorm1d(512,momentum=0,track_running_stats=False),
                    nn.LeakyReLU(),
                    )

        self.proj_mat = nn.Linear(512, self.d_e, bias=False)
        self.to(self.device)

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

        self.conv=nn.Sequential(
                nn.BatchNorm2d(1,momentum=0,track_running_stats=False),
                # Initial glove features
                nn.Conv2d(1,512,(1, GLOVE_DIM),padding=(0,0), bias=False),
                # 3 (in the 15 ms case) x 1 
                #nn.BatchNorm2d(512,momentum=.1),
                nn.BatchNorm2d(512, momentum=0, track_running_stats=False),
                nn.LeakyReLU(),
                nn.Flatten(),

                nn.Linear(512, 512, bias=False),
                nn.BatchNorm1d(512, momentum=0, track_running_stats=False),
                nn.LeakyReLU(),
                nn.Dropout(p=.5),
                
                nn.Linear(512, 512, bias=False),
                nn.BatchNorm1d(512, momentum=0, track_running_stats=False),
                nn.LeakyReLU(),
                nn.Dropout(p=.5),

                nn.Linear(512, 512//2, bias=False),
                nn.BatchNorm1d(512//2, momentum=0, track_running_stats=False),
                nn.LeakyReLU(),
                )

        self.proj_mat = nn.Linear(512//2, self.d_e, bias=False)
        
        self.to(self.device)

    def forward(self, GLOVE):
        #GLOVE=GLOVE.mean(dim=2, keepdim=True)
        GLOVE=GLOVE[:, :, :1]
        out=self.conv(GLOVE)
        out=self.proj_mat(out)
        return out

    def l2(self):
        reg_loss = 0
        for name,param in self.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                reg_loss+=torch.norm(param)
        return reg_loss

