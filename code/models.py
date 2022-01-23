import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from constants import *
from utils import *
from torch.nn.modules.utils import _pair
import ipdb

#torch.autograd.set_detect_anomaly(True)

# adaptive batch normalization - https://doi.org/10.1016/j.patcog.2018.03.005
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.determinist=True

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

class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out

# modeled after https://github.com/openai/CLIP/blob/main/clip/model.py
class Model(nn.Module):
    def __init__(self, params, adabn=True, train_model=True, prediction=False, glove=False, device="cuda"):
        super(Model,self).__init__()

        self.params=params
        self.train_model = train_model
        self.adabn=adabn
        self.prediction=prediction
        self.glove=glove
        self.device = torch.device(device)

        self.emg_net = EMGNet(d_e=params['d_e'], dp=params['dp_emg'], adabn=adabn, prediction=prediction) 
        self.glove_net = GLOVENet(d_e=params['d_e'], dp=params['dp_glove'], adabn=adabn, prediction=prediction) 

        self.loss_f = torch.nn.functional.cross_entropy
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1)/0.07)
        self.to(self.device)

        self.correct_tr = []
        self.correct_v = []

    def set_train(self):
        self.train_model=True
        self.train()
        self.reset()

    def set_test(self):
        self.train_model=False
        self.eval()
        self.reset()

    def set_val(self):
        self.set_test() # same behavior

    def reset(self):
        self.corrects = []

    def encode_emg(self, EMG):
        return self.emg_net(EMG)

    def encode_glove(self, GLOVE):
        return self.glove_net(GLOVE)

    def forward(self, EMG, GLOVE):
        #ipdb.set_trace()
        if self.prediction:
            if self.glove:
                features = self.encode_glove(GLOVE)
            else:
                features = self.encode_emg(EMG)
            features = features / features.norm(dim=-1,keepdim=True)
            return features
        else:
            emg_features = self.encode_emg(EMG)
            emg_features = emg_features / emg_features.norm(dim=-1,keepdim=True)
            glove_features = self.encode_glove(GLOVE)
            glove_features = glove_features / glove_features.norm(dim=-1,keepdim=True)
            # becomes (N, d_e, tasks)
            glove_features_t = glove_features.transpose(1,2)
            # (N, tasks_e, d_e) x (N, d_e, tasks_g) -> (N, tasks_e, tasks_g)
            logits=torch.bmm(emg_features * 100, glove_features_t * 100) * self.logit_scale
            return logits

    def contrastive_loopy_loss(self, logits, labels, acc=False):
        loss=torchize([0])
        correct=torchize([0]).to(torch.float)
        for log in logits:
            loss = loss + self.loss_f(log, labels[:log.shape[0]])
            if acc:
                correct += (F.softmax(log, dim=-1).detach().cpu().numpy().argmax(-1)==labels[:log.shape[0]].cpu().numpy()).mean()
        loss=loss/np.prod(logits.shape[0])
        if acc:
            # correct for the values we want (predicting grasp from emg)
            correct=correct/logits.shape[0]
            self.corrects.append(correct.item())
        return loss

    def prediction_loss(self, logits, labels):
        loss=self.loss_f(logits, labels)
        correct = (F.softmax(logits, dim=-1).detach().cpu().numpy().argmax(-1)==labels.cpu().numpy()).mean()
        self.corrects.append(correct.item())
        return loss
        
    def loss(self, logits, labels):
        l=(labels.cpu().numpy())
        if self.prediction:
            loss=self.prediction_loss(logits, labels)
        else:
            # loopy-loopy first, then vectorized
            loss_e=self.contrastive_loopy_loss(logits, labels, acc=True)
            # tasks_e x tasks_g -> tasks_g x tasks_e
            loss_g=self.contrastive_loopy_loss(torch.transpose(logits,1,2), labels)
            loss=(loss_e+loss_g)/2
        return loss

    def correct(self):
        return np.array(self.corrects).mean()

    def l2(self):
        if self.prediction:
            return self.glove_net.l2()*self.params['reg_glove'] if self.glove else self.emg_net.l2()*self.params['reg_emg'] 
        return self.glove_net.l2()*self.params['reg_glove'] + self.emg_net.l2()*self.params['reg_emg'] 

class EMGNet(nn.Module):
    def __init__(self, d_e, dp=.5, adabn=True, train=True, prediction=False, device="cuda"):
        super(EMGNet,self).__init__()
        self.device=torch.device(device)
        self.d_e=d_e
        self.dp=dp
        self.prediction=prediction

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
                nn.Conv2d(1,64,(1,3),padding=(0,1),bias=False),
                self.bn2d_func(64),
                nn.ReLU(),

                nn.Conv2d(64,64,(1,3),padding=(0,1),bias=False),
                self.bn2d_func(64),
                nn.ReLU(),

                )

        self.local = nn.Sequential(
                LocallyConnected2d(64, 32, (1,EMG_DIM), 1, 1, bias=False),
                self.bn2d_func(32),
                nn.ReLU(),

                LocallyConnected2d(32, 32, (1,EMG_DIM),1, 1, bias=False),
                self.bn2d_func(32),
                nn.ReLU(),
                nn.Dropout(self.dp),

                nn.Flatten(),
            )
        
        self.linear = nn.Sequential(
                nn.Linear(32*EMG_DIM, 512,bias=False),
                self.bn1d_func(512),
                nn.ReLU(),
                nn.Dropout(self.dp),

                nn.Linear(512, 512,bias=False),
                self.bn1d_func(512),
                nn.ReLU(), 
                nn.Dropout(self.dp),
                )

        self.simple = nn.Sequential(
                nn.Flatten(),
                nn.Linear(EMG_DIM, 64),
                nn.ReLU(),
                nn.Linear(64, MAX_TASKS_TRAIN, bias=False)
                )

        if self.prediction:
            self.last = nn.Sequential(
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    self.bn1d_func(128),
                    #nn.Dropout(self.dp),

                    nn.Linear(128, MAX_TASKS_TRAIN, bias=False),
                    )
        else:
            self.last = nn.Sequential(
                    # projection
                    nn.Linear(512, self.d_e, bias=False),
                    )


        self.to(self.device)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, LocallyConnected2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        
    def forward(self, EMG):
        out=EMG.reshape(-1, 1, 1, EMG_DIM)
        #out=self.simple(out)
        out=self.conv_emg(out)
        out=self.local(out)
        out=self.linear(out)
        if not self.prediction:
            # reshape back
            out=out.reshape((EMG.shape[0], EMG.shape[1], -1))
        out=self.last(out)
        return out

    def l2(self):
        reg_loss = 0
        for name,param in self.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                reg_loss+=torch.norm(param)
        return reg_loss


class GLOVENet(nn.Module):
    def __init__(self, d_e, dp=.5, adabn=True, train=True, prediction=False, device="cuda"):
        super(GLOVENet,self).__init__()
        self.device=torch.device(device)
        self.d_e=d_e
        self.dp=dp
        self.prediction=prediction
        if adabn:
            self.bn1d_func = AdaBatchNorm1d
            self.bn2d_func = AdaBatchNorm2d
        else:
            self.bn1d_func = nn.BatchNorm1d
            self.bn2d_func = nn.BatchNorm2d

        # momentum = 0 and batch per subject in order to have adaptive normalization (https://doi.org/10.1016/j.patcog.2018.03.005)
        # loosely inspired by architecture in https://doi.org/10.3390/s17030458

        self.conv_glove=nn.Sequential(
                # conv -> bn -> relu
                #self.bn2d_func(1),

                nn.Conv2d(1,64,(1,3),padding=(0,1)),
                nn.ReLU(),
                self.bn2d_func(64),

                nn.Conv2d(64,64,(1,3),padding=(0,1)),
                nn.ReLU(),
                self.bn2d_func(64),

                nn.Flatten(),
                )

        self.linear = nn.Sequential(
                nn.Flatten(),

                nn.Linear(GLOVE_DIM*64, 512//2),
                nn.ReLU(),
                self.bn1d_func(512//2),

                nn.Linear(512//2, 512//2),
                nn.ReLU(),
                self.bn1d_func(512//2),
                nn.Dropout(self.dp),

                nn.Linear(512//2, 512//2),
                nn.ReLU(), 
                self.bn1d_func(512//2),
                nn.Dropout(self.dp),
                )

        if self.prediction:
            self.last = nn.Sequential(
                    nn.Linear(512//2, 128),
                    nn.ReLU(),
                    self.bn1d_func(128),
                    nn.Dropout(self.dp),

                    nn.Linear(128, MAX_TASKS_TRAIN, bias=False),
                    )
        else:
            self.last = nn.Sequential(
                    # projection
                    nn.Linear(512, self.d_e, bias=False),
                    )

        self.to(self.device)

    def forward(self, GLOVE):
        out=GLOVE.reshape(-1, 1, 1, GLOVE_DIM)
        out=self.conv_glove(out)
        out=self.linear(out)
        if not self.prediction:
            # reshape back
            out=out.reshape((GLOVE.shape[0], GLOVE.shape[1], -1))
        out=self.last(out)
        return out

    def l2(self):
        reg_loss = 0
        for name,param in self.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                reg_loss+=torch.norm(param)
        return reg_loss
