import numpy as np
import torch
import math
from constants import *
from scipy.ndimage import uniform_filter1d
from scipy import signal
import scipy.io as sio

def torchize(X):
    return torch.from_numpy(np.array(X)).to(torch.device("cuda"))

class Sampler():
    def __init__(self, dataset, batch_size):
        self.dataset=dataset
        self.bs=batch_size
        self.device=torch.device("cuda")
        self.tasks=dataset.TASKS
        self.d=len(dataset)//self.tasks
        self.num_points = self.d // self.bs
        self.left_overs = self.d % self.bs
        print(self.num_points, self.left_overs)
        self.reset()

    def reset(self):
        self.rand_idxs = torch.empty((self.tasks, self.d), device=self.device, dtype=torch.long)
        for t in range(self.tasks):
            self.rand_idxs[t] = torch.randperm(self.d, dtype=torch.long, device=self.device)+self.d*t
        # tasks x d -> d x tasks -> bs*tasks x -1
        self.rand_idxs = self.rand_idxs.T[:-self.left_overs].flatten().reshape(self.num_points, self.bs*self.tasks)

    def __len__(self):
        return self.num_points

    def __iter__(self):
        for batch in self.rand_idxs:
            yield batch
        self.reset()
        return

# https://www.johndcook.com/blog/standard_deviation/
class RunningStats():
    def __init__(self, norm=False, complete=False):
        self.counter = 0
        # if norm, then normalize with mean and std
        self.norm=norm
        self.complete = complete
        if not norm:
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
        emg,  stim, rep = Es[ex]
        eh,el=np.percentile(emg, [99.5, .5], axis=0)
        emg=remove_outliers(emg, EMG_DIM, low=el, high=eh, factor=1)
        Es_new.append((emg, stim, rep))
    return tuple(Es_new)

def get_np(dbnum, p_dir, n):
    E_mat=sio.loadmat("../db%s/s%s/S%s_E%s_A1"%(dbnum,p_dir,p_dir,n))
    emg=E_mat['emg'] # 12
    stimulus=E_mat['restimulus']
    repetition=E_mat['rerepetition']
    return emg, stimulus, repetition

