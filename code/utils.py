import numpy as np
import torch
import math
from constants import *
from scipy.ndimage import uniform_filter1d
from scipy import signal
import scipy.io as sio
from tqdm import tqdm

import line_profiler, builtins, atexit
profile=line_profiler.LineProfiler()
atexit.register(profile.print_stats)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

def torchize(X):
    return torch.from_numpy(np.array(X)).to(torch.device("cuda"))

class TaskWrapper():
    def __init__(self, dataset):
        self.dataset=dataset
        self.device=torch.device("cuda")
        # you need to set_....() in order to make it create rand
        #self.reset()
        
    def return_rand(self, D):
        self.rand = torch.empty((self.dataset.TASKS, D), device=self.device, dtype=torch.long)
        for t in range(self.dataset.TASKS):
            self.rand[t] = torch.randperm(D, dtype=torch.long, device=self.device)+D*t
        return self.rand

    def reset(self):
        self.emg_rand=self.return_rand(self.dataset.D)
        self.glove_rand=self.return_rand(self.dataset.glover.D)
        self.idx=torch.randperm(self.dataset.TASKS*self.dataset.D, device=self.device, dtype=torch.long)

    def __getattr__(self, name):
        #def method(*args, **kwargs):
        return getattr(self.dataset, name)
        #return method

    def __len__(self):
        return self.dataset.D

    def __getitem__(self, idx):
        tensor_emg = self.dataset[self.emg_rand[:, idx]]
        tensor_glove = self.dataset.glover[self.glove_rand[:, idx%self.dataset.glover.D]]
        label=torch.arange(self.dataset.TASKS, device=self.device, dtype=torch.long)

        #idxx=self.idx[self.dataset.TASKS*idx:self.dataset.TASKS*(idx+1)]
        #tensor_emg=self.dataset[idxx]
        #tensor_glove = self.dataset.glover[idxx]
        #label=(idxx//self.dataset.D).to(torch.long).flatten()

        # move to half precision
        tensor_emg=tensor_emg.to(torch.float32)
        tensor_glove=tensor_glove.to(torch.float32)
        return tensor_emg, tensor_glove, label

    def set_train(self):
        self.dataset.set_train()
        self.reset()

    def set_val(self):
        self.dataset.set_val()
        self.reset()

    def set_test(self):
        self.dataset.set_test()
        self.reset()

# https://www.johndcook.com/blog/standard_deviation/
class RunningStats():
    def __init__(self, path_dir, complete=False):
        self.counter = 0
        self.complete = complete
        self.path=path_dir

    def push(self, X):
        self.counter+=1
        self.np = isinstance(X, np.ndarray)

        X=X.mean(0) # along window size dimension (constant for all X)
        if self.counter==1:
            self.old_mean = self.new_mean = X
            self.old_std = np.zeros(X.shape, dtype=X.dtype) if self.np else torch.zeros(X.shape, device=X.device, dtype=X.dtype)
        else:
            self.new_mean=self.old_mean + (X-self.old_mean)/self.counter
            self.new_std=self.old_std+(X-self.old_mean)*(X-self.new_mean)
            self.old_mean=self.new_mean
            self.old_std=self.new_std

    def mean(self):
        mean=self.new_mean
        if self.complete:
            mean=mean.mean()
        if not self.np:
            np.save(self.path+"mean.npy", mean.cpu().numpy())
        else:
            np.save(self.path+"mean.npy", mean)
        return mean

    def variance(self):
        return (self.new_std)/(self.counter-1)

    def std(self):
        var = self.variance()
        if self.complete:
            var = var.mean()
        if self.np:
            std=np.sqrt(self.variance())
        else:
            std=torch.sqrt(self.variance())
        if not self.np:
            np.save(self.path+"std.npy", std.cpu().numpy())
        else:
            np.save(self.path+"std.npy", std)
        return std

    def mean_std(self):
        return self.mean(), self.std()

    def normalize(self, X):
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

class MinMaxTracker():
    def __init__(self):
        self.min=np.inf
        self.max=-np.inf

    def push(self, val):
        self.min=min(val, self.min)
        self.max=max(val, self.max)

class Glover():
    def __init__(self):
        self.path="/home/breezy/Downloads/"
        # from 28 to 68
        self.GLOVE_PEOPLE=np.arange(28,67,dtype=np.uint8)
        # for the minimum length of the thing
        self.task_cumsum=TASK_DIST.cumsum()
        self.device=torch.device("cuda")
        self.angle_idxs=np.arange(22, dtype=np.uint8)
        # no angle index 5 -> nans, or 10 -> noisy
        self.angle_idxs=np.delete(self.angle_idxs, [5,10])

    def get_np(self, p_dir, n):
        E_mat=sio.loadmat("/home/breezy/Downloads/s_%s_angles/S%s_E%s_A1"%(p_dir,p_dir,n))
        angles=E_mat['angles'][:, self.angle_idxs]
        stimulus=E_mat['restimulus']
        repetition=E_mat['rerepetition']
        return angles, stimulus, repetition

    def get_person_dat(self, person):
        self.E1=self.get_np(str(person+1), "1")
        self.E2=self.get_np(str(person+1), "2")
        self.Es=(self.E1,self.E2)

    def get_task(self, stim):
        ex=np.searchsorted(self.task_cumsum, stim)
        angles,stimulus,repetition=self.Es[ex]
        mask=(stimulus==stim)
        reps_angles=[angles[(mask&(repetition==rep)).flatten()][:GLOVE_WINDOW_SIZE] for rep in range(1,repetition.max()+1)]
        angles=np.concatenate(np.array(reps_angles),axis=0)
        return angles

    def save(self):
        torch.save(torchize(self.GLOVE), PATH_DIR+'data/glove.pt')

    def load_stored(self):
        self.GLOVE=torch.load(PATH_DIR+'data/glove.pt', map_location=self.device)
        self.GLOVE.cuda()
        print("Loading stored glove...", self.GLOVE.shape)
        return self.GLOVE

    def load_dataset(self):
        dats=[]
        self.stats = RunningStats(PATH_DIR+"data/glove_")

        for person in tqdm(self.GLOVE_PEOPLE):
            self.get_person_dat(person)
            all_tasks=[]
            for stim in range(MAX_TASKS):
                dat=self.get_task(stim)
                all_tasks.append(dat)
            all_tasks=np.array(all_tasks)
            dats.append(all_tasks)
            self.stats.push(all_tasks[TRAIN_TASKS].reshape(-1,GLOVE_DIM))

        self.GLOVE = np.concatenate(dats, axis=1)
        print("Glove shape:", self.GLOVE.shape)
        print("Normalizing glove...")
        self.GLOVE=self.stats.normalize(self.GLOVE)
        print("Glove normalized.")
        self.save()
        print("Glove (un)loaded successfully")

    def load_valid(self, tasks_mask):
        tensor=self.GLOVE[tasks_mask]
        self.D=self.GLOVE.shape[1]
        self.GLOVE_use=tensor.reshape(-1, GLOVE_DIM)

    def __getitem__(self, idx):
        return self.GLOVE_use[idx]
