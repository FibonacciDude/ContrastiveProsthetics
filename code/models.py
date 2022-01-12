import torch
import torch.nn as nn
import numpy as np
from constants import *
from utils import RunningStats

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
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1)/0.07)    # CLIP logit scale
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
        emg_features = emg_features / emg_features.norm(dim=-1,keepdim=True)
        glove_features = glove_features / glove_features.norm(dim=-1,keepdim=True)
        #print(emg_features.norm(dim=1,keepdim=True).shape)
        #print(emg_features.shape, glove_features.shape)
        emg_features = emg_features.reshape((EMG_T,-1,self.d_e)).permute((1,0,2))
        glove_features = glove_features.reshape((GLOVE_T,-1,self.d_e)).permute((1,0,2))
        
        #                        -> N_e x N_g
        # encoders give input as (TASKS*WINDOW_BLOCK, d_e) or another N
        # we want (-1, TASKS, d_e) and take cross entropy across entire (-1) dim

        logit_scale=self.logit_scale.exp().clamp(min=1e-8,max=100)
        #print(emg_features.shape)
        logits = torch.matmul(emg_features, glove_features.permute(0,2,1)) # shape = (N,TASKS_e,TASKS_g)

        if self.train_model:
            loss= self.loss(logits * logit_scale)
            logits = logits * logit_scale
            return loss, logits
        return logits

    def loss(self, logits):
        # matrix should be symmetric
        N,tasks,tasks=logits.shape  # e x g
        labels = self.get_labels(N,tasks)
        #print(logits.shape)
        #print(labels)
        #print(logits.shape)
        # convert (N_e, N_g) -> (n,task_e,N_g) -> (n,task_e,n,task_g) -> (n,n,task_g,task_e) -> (n^2,task_g,task_e)
        #print(logits.shape)
        logits_e = logits.reshape((N*tasks,tasks))
        logits_g = logits.permute((0,2,1)).reshape((N*tasks,tasks))
        loss_e = self.loss_f(logits_e, labels, reduction='mean')
        loss_g = self.loss_f(logits_g, labels, reduction='mean')
        loss = (loss_e+loss_g)/2
        #print(loss)
        return loss

    def get_labels(self, N, tasks):
        return torch.stack([torch.arange(tasks,dtype=torch.long,device=self.device)]*N).reshape(N*tasks)

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

    def l2(self):
        return self.emg_net.l2() + self.glove_net.l2()

class EMGNet(nn.Module):
    def __init__(self, d_e, train=True, device="cuda"):
        super(EMGNet,self).__init__()
        self.device=torch.device(device)
        self.d_e=d_e

        # No fusion for now. 
        # Reason: The classification might be affected with different activites
        # and the acceleration data shifts for different windows (relevant to static)
        self.use_acc = False


        # momentum = 0 and batch per subject in order to have adaptive normalization (https://doi.org/10.1016/j.patcog.2018.03.005)

        # loosely inspired by architecture in https://doi.org/10.3390/s17030458

        self.conv_emg=nn.Sequential(
                # conv -> bn -> relu
                AdaBatchNorm2d(1),

                # prevent premature fusion (https://www.mdpi.com/2071-1050/10/6/1865/htm) 
                # larger kernel
                nn.Conv2d(1,32,(3,3),padding=(1,1), bias=False),
                nn.LeakyReLU(),
                AdaBatchNorm2d(32),

                # smaller kernel
                nn.Conv2d(32,64,(3,3),padding=(1,1), bias=False),
                nn.LeakyReLU(),
                AdaBatchNorm2d(64),

                nn.Conv2d(64,64,(1,1), bias=False),
                nn.LeakyReLU(),
                AdaBatchNorm2d(64),

                # added
                nn.Conv2d(64, 64, (3, 3),padding=(1,1), bias=False),
                nn.LeakyReLU(),
                AdaBatchNorm2d(64),

                nn.Conv2d(64,64,(1,1),padding=(0,0),bias=False),
                nn.LeakyReLU(),
                AdaBatchNorm2d(64),
                nn.Dropout(.5),
                
                nn.Conv2d(64,64,(1,1), bias=False),
                nn.LeakyReLU(),
                AdaBatchNorm2d(64),
                nn.Dropout(.5),
                nn.Flatten(),

                nn.Linear(EMG_DIM*WINDOW_MS*64, 512, bias=False),
                nn.LeakyReLU(),
                #AdaBatchNorm1d(512)
                )

        # no acceleration data - can vary depending on activity of person
        # with prosthesis (like walking)
        self.proj_mat = nn.Linear(512, self.d_e, bias=False)
        self.to(self.device)

    def forward(self, EMG, ACC):
        ACC=ACC.squeeze(1)
        #EMG=torch.randn(EMG.shape).to(torch.device("cuda"))
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

        # Maybe the adaptive normalization has to be in the first (or few) layers
        # and normal afterwards (I don't think we have to normalize that much really...

        self.conv=nn.Sequential(
                AdaBatchNorm2d(1),
                # Initial glove features
                nn.Conv2d(1,32,(1,3),padding=(0,1), bias=False),
                # 3 (in the 15 ms case) x 1 
                nn.LeakyReLU(),
                AdaBatchNorm2d(32),

                nn.Conv2d(32, 64, (1,3), padding=(0,1), bias=False),
                nn.LeakyReLU(),
                AdaBatchNorm2d(64),

                # added 
                nn.Conv2d(64, 64, (1,1), padding=(0,0), bias=False),
                nn.LeakyReLU(),
                AdaBatchNorm2d(64),
                #nn.Dropout(p=.5),
                
                nn.Conv2d(64, 64, (1,1), padding=(0,0), bias=False),
                nn.LeakyReLU(),
                AdaBatchNorm2d(64),
                nn.Dropout(p=.5),

                nn.Flatten(),
                nn.Linear(64*GLOVE_DIM, 256, bias=False),
                nn.LeakyReLU(),
                AdaBatchNorm1d(256),
                nn.Dropout(p=.5),
                
                nn.Linear(256, 256, bias=False),
                nn.LeakyReLU(),
                AdaBatchNorm1d(256),

                # Should you have these before z vector?  (dropout + bn)
                # Look at SOTA contrastive models and copy
                )

        self.proj_mat = nn.Linear(256, self.d_e, bias=False)
        
        self.to(self.device)

    def forward(self, GLOVE):
        #GLOVE=GLOVE.mean(dim=2, keepdim=True)
        GLOVE=GLOVE[:, :, :1]
        #GLOVE=torch.randn(GLOVE.shape).to(torch.device("cuda"))
        out=self.conv(GLOVE)
        out=self.proj_mat(out)
        return out

    def l2(self):
        reg_loss = 0
        for name,param in self.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                reg_loss+=torch.norm(param)
        return reg_loss

