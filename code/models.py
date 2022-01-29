import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from constants import *
from utils import *
import ipdb

#torch.autograd.set_detect_anomaly(True)

# adaptive batch normalization - https://doi.org/10.1016/j.patcog.2018.03.005
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
#torch.backends.cudnn.deterministic=True

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
            logits=torch.bmm(emg_features, glove_features_t) * self.logit_scale
            return logits

    def contrastive_loopy_loss(self, logits, labels, acc=False):
        loss=torchize([0])
        correct=torchize([0]).to(torch.float)

        shape=self.emg_net.shape
        if not self.training and VOTE:
            logits=logits.reshape((shape[0], shape[2], shape[1], shape[1]))
            times=shape[2]
        else:
            times=1

        for log in logits:
            loss = loss + self.loss_f(log.reshape(-1, shape[1]), torch.cat([labels[:log.shape[-1]]]*times)   )
            if acc:
                pred=F.softmax(log, dim=-1).argmax(-1)
                if not self.training and VOTE:
                    pred=pred.mode(0)[0]
                equal=(pred==labels[:log.shape[-1]]).cpu().numpy()
                correct += equal.mean()
        loss=loss/np.prod(logits.shape[0])
        if acc:
            # correct for the values we want (predicting grasp from emg)
            correct=correct/logits.shape[0]
            self.corrects.append(correct.item())
        return loss

    def prediction_loss(self, logits, labels):
        if not self.training and VOTE and not self.glove:
            shape=logits.shape
            assert len(shape)==3, "wrong logit shape for val time"
            labels_=labels.reshape(-1, labels.max()+1, 1).expand(-1, labels.max()+1, logits.shape[1]).flatten().to(torch.long)
            logits_=logits.reshape(-1, self.emg_net.bits)
        else:
            labels_=labels
            logits_=logits
        loss=self.loss_f(logits_, labels_)

        prediction=F.softmax(logits, dim=-1).argmax(-1)
        if self.training or not VOTE or self.glove:
            correct = (prediction.detach().cpu().numpy()==labels_.cpu().numpy())
        else:
            # majority voting in action
            maj_vote=prediction.mode(1)[0]
            correct = (maj_vote.detach().cpu().numpy()==labels.cpu().numpy())
        correct = correct.mean()

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

                nn.Linear(512, 512),
                nn.ReLU(),
                self.bn1d_func(512),
                nn.Dropout(self.dp),

                nn.Linear(512, 512),
                nn.ReLU(), 
                self.bn1d_func(512),
                nn.Dropout(self.dp),
                )

        if self.prediction:
            self.bits=MAX_TASKS_TRAIN
            self.last = nn.Sequential(
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    self.bn1d_func(128),
                    nn.Dropout(self.dp),

                    nn.Linear(128, self.bits, bias=False),
                    )
        else:
            self.bits=self.d_e
            self.last = nn.Sequential(
                    # projection
                    nn.Linear(512, self.d_e, bias=False),
                    )

        self.to(self.device)

    def forward(self, EMG):
        self.shape=EMG.shape
        out=EMG.reshape(-1, 1, 1, EMG_DIM)
        out=self.conv_emg(out)
        out=self.linear(out)
        out=self.last(out)
        vote=not self.training and VOTE
        shape=self.shape
        if vote:
            if self.prediction:
                out=out.reshape((-1, shape[2], self.bits))
            else:
                out=out.reshape((shape[0], shape[1], shape[2], self.bits))
                out=out.transpose(1,2)  # 32,20,38,bits
                out=out.reshape((-1, shape[1], self.bits))
        if not self.prediction:
            out=out.reshape((-1, shape[1], self.bits))
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

                #nn.Conv2d(1,64,(1,3),padding=(0,1)),
                #nn.ReLU(),
                #self.bn2d_func(64),

                #nn.Conv2d(64,64,(1,3),padding=(0,1)),
                #nn.ReLU(),
                #self.bn2d_func(64),

                nn.Flatten(),
                )

        self.linear = nn.Sequential(
                nn.Flatten(),
                nn.Linear(GLOVE_DIM, 10),
                nn.Linear(10, GLOVE_DIM),

                nn.Linear(GLOVE_DIM, 512//2),
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

                nn.Linear(512//2, 512//2),
                nn.ReLU(), 
                self.bn1d_func(512//2),
                nn.Dropout(self.dp),
                )

        self.bits=MAX_TASKS_TRAIN if self.prediction else self.d_e
        if self.prediction:
            self.last = nn.Sequential(
                    nn.Linear(512//2, 128),
                    nn.ReLU(),
                    self.bn1d_func(128),
                    nn.Dropout(self.dp),

                    nn.Linear(128, self.bits, bias=False),
                    )
        else:
            self.last = nn.Sequential(
                    # projection
                    nn.Linear(512//2, self.bits, bias=False),
                    )

        self.to(self.device)

    def forward(self, GLOVE):
        out=GLOVE.reshape(-1, 1, 1, GLOVE_DIM)
        out=self.conv_glove(out)
        out=self.linear(out)
        out=self.last(out)
        vote=not self.training and VOTE
        if not self.prediction:
            # reshape back
            out=out.reshape((GLOVE.shape[0], -1, self.bits))
            shape=out.shape
            if vote:
                out=out.reshape((shape[0], 1, shape[1], shape[2])).expand(-1, PREDICTION_WINDOW_SIZE, -1, -1).reshape(-1, shape[1], self.bits)
        return out

    def l2(self):
        reg_loss = 0
        for name,param in self.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                reg_loss+=torch.norm(param)
        return reg_loss
