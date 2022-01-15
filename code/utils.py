import numpy as np
import torch
from constants import *
from scipy.ndimage import uniform_filter1d
from scipy import signal
import scipy.io as sio

# https://www.johndcook.com/blog/standard_deviation/

class RunningStats():
    def __init__(self):
        self.counter = 0

    def push(self, X):
        self.counter+=1
        X=X.mean(0) # along window size dimension (constant for all X)
        if self.counter==1:
            self.old_mean = self.new_mean = X
            self.np = isinstance(X, np.ndarray)
            if self.np:
                self.old_std = np.zeros(X.shape, dtype=X.dtype)
            else:
                self.old_std = torch.zeros(X.shape, device=X.device, dtype=X.dtype)
        else:
            self.new_mean=self.old_mean + (X-self.old_mean)/self.counter
            self.new_std=self.old_std+(X-self.old_mean)*(X-self.new_mean)
            self.old_mean=self.new_mean
            self.old_std=self.new_std

    def mean(self):
        return self.new_mean

    def variance(self):
        return (self.new_std)/(self.counter-1)

    def std(self):
        if self.np:
            return np.sqrt(self.variance())
        return torch.sqrt(self.variance())

    def mean_std(self):
        return self.mean(), self.std()


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
        transpose[i] = (signal.filtfilt(b, a, transpose[i]))
    return transpose.T

# https://github.com/Answeror/semimyo/blob/master/sigr/data/preprocess.py
# moving window rms
def moving_rms(data):
    # take out invalid values
    return np.sqrt(uniform_filter(np.square(data), size=RMS_WINDOW, mode='nearest'))[WINDOW_EDGE:-WINDOW_EDGE]

def rms(data):
    transpose = data.T
    for i in range(len(transpose)):
        transpose[i] = moving_rms(transpose[i])
    return transpose.T


def remove_outliers(tensor, dim, low, high, factor=1):
    for d in range(dim):
        tensor.T[d]=np.clip(tensor.T[d], a_min=low[d], a_max=high[d])
    return tensor

def get_np(dbnum, p_dir, n):
    E_mat=sio.loadmat("../db%s/s%s/S%s_E%s_A1"%(dbnum,p_dir,p_dir,n))
    emg=E_mat['emg'] # 12
    acc=E_mat['acc'] # 36
    glove=E_mat['glove'] # 22
    stimulus=E_mat['restimulus']
    repetition=E_mat['rerepetition']
    return emg, acc, glove, stimulus, repetition

