import numpy as np
import torch
import math
from constants import *
from scipy.ndimage import uniform_filter1d
from scipy import signal
import scipy.io as sio

def torchize(X):
    return torch.from_numpy(np.array(X)).to(torch.device("cuda"))

# https://www.johndcook.com/blog/standard_deviation/
class RunningStats():
    def __init__(self, norm=False, complete=False):
        self.counter = 0
        # if norm, then normalize with mean and std
        self.norm=norm
        self.complete = complete
        if not norm:
            #self.min = float('inf')
            #self.max = -float('inf')
            self.min = 10**10
            self.max = -10**10

    def push(self, X):
        self.counter+=1
        self.np = isinstance(X, np.ndarray)
        if not self.np and self.counter==1 and not self.norm:
            self.min=torchize(self.min)
            self.max=torchize(self.max)
        
        if self.norm:
            X=X.mean(0) # along window size dimension (constant for all X)
            if self.counter==1:
                self.old_mean = self.new_mean = X
                self.old_std = np.zeros(X.shape, dtype=X.dtype) if self.np else torch.zeros(X.shape, device=X.device, dtype=X.dtype)
            else:
                self.new_mean=self.old_mean + (X-self.old_mean)/self.counter
                self.new_std=self.old_std+(X-self.old_mean)*(X-self.new_mean)
                self.old_mean=self.new_mean
                self.old_std=self.new_std
        else:
            if self.np:
                x_max,x_min=np.percentile(X, [99.5, .5])
                self.min = np.min(self.min, x_min)
                self.max = np.max(self.max, x_max)
            else:
                x_max,x_min=np.percentile(X.cpu().numpy(), [99.5, .5])
                self.min = torch.minimum(self.min, torchize(x_min))
                self.max = torch.maximum(self.max, torchize(x_max))

    def mean(self):
        mean=self.new_mean
        if self.complete:
            mean=mean.mean()
        return mean

    def variance(self):
        return (self.new_std)/(self.counter-1)

    def std(self):
        var = self.variance()
        if self.complete:
            var = var.mean()
        if self.np:
            return np.sqrt(self.variance())
        return torch.sqrt(self.variance())

    def mean_std(self):
        return self.mean(), self.std()

    def min_max(self):
        return self.min, self.max

    def normalize(self, X):
        if not self.norm:
            return (X-self.min)/(self.max-self.min)
        return (X-self.mean())/self.std()


# credit to github user parasgulati8
def filter(data, f, butterworth_order=4,btype="bandpass"):
    nyquist=Hz/2
    if isinstance(f, int):
        fc = f/nyquist
    else:
        fc = list(f)
        for i in range(len(f)):
            fc[i] = fc[i]/nyquist
    b,a = signal.butter(butterworth_order, fc, btype=btype)
    transpose = data.T
    
    for i in range(len(transpose)):
        transpose[i] = (signal.lfilter(b, a, transpose[i]))
    return transpose.T

# https://github.com/Answeror/semimyo/blob/master/sigr/data/preprocess.py
# moving window rms
def moving_rms(data):
    # take out invalid values
    return np.sqrt(uniform_filter1d(np.square(data), size=RMS_WINDOW, mode='nearest'))[WINDOW_EDGE:-WINDOW_EDGE]

def rms(data):
    return np.transpose([moving_rms(t) for t in data.T])

def remove_outliers(tensor, dim, low, high, factor=1):
    for d in range(dim):
        tensor.T[d]=np.clip(tensor.T[d], a_min=low[d], a_max=high[d])
    return tensor

def clip_es(Es):
    Es_new = []
    for ex in range(2):
        emg, acc, glove, stim, rep = Es[ex]
        eh,el=np.percentile(emg, [99.5, .5], axis=0)
        ah,al=np.percentile(acc, [99.5, .5], axis=0)
        gh,gl=np.percentile(glove, [99.5, .5], axis=0)
        glove=remove_outliers(glove, GLOVE_DIM, low=gl, high=gh, factor=1)
        acc=remove_outliers(acc, ACC_DIM, low=al, high=ah, factor=1)
        emg=remove_outliers(emg, EMG_DIM, low=el, high=eh, factor=1)
        Es_new.append((emg, acc, glove, stim, rep))
    return tuple(Es_new)

def get_np(dbnum, p_dir, n):
    E_mat=sio.loadmat("../db%s/s%s/S%s_E%s_A1"%(dbnum,p_dir,p_dir,n))
    emg=E_mat['emg'] # 12
    acc=E_mat['acc'] # 36
    glove=E_mat['glove'] if not (n=="3") else None
    stimulus=E_mat['restimulus']
    repetition=E_mat['rerepetition']
    return emg, acc, glove, stimulus, repetition

