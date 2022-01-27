import numpy as np
import torch
import math
from constants import *
from scipy.ndimage import uniform_filter1d
from scipy import signal
import scipy.io as sio
from tqdm import tqdm
import torch
import line_profiler, builtins, atexit
import torch.utils.data as data

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

def torchize(X):
    return torch.from_numpy(np.array(X)).to(torch.device("cuda"))

class BatchSampler(data.Sampler):
    def __init__(self, dataset, batch_size, drop_last=False):
        self.dt = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.device=torch.device("cuda")

    def reset(self):
        b=torch.arange(self.dt.PEOPLE, device=self.device, dtype=torch.long).reshape(self.dt.PEOPLE,1)*self.dt.D
        self.idxs=torch.rand((self.dt.PEOPLE, self.dt.D), device=self.device).argsort(dim=-1)+b
        self.randperm=torch.randperm(self.__len__(), device=self.device, dtype=torch.long)

    def __iter__(self):
        self.reset()
        for idx_raw in self.randperm:
            people_idx=idx_raw // self.amt_batches
            else_idx=idx_raw % self.amt_batches
            batch_idxs=self.idxs[people_idx][else_idx*self.batch_size:(else_idx+1)*self.batch_size]
            batches=[self.dt[batch_idx] for batch_idx in batch_idxs]
            emg,glove,labels=zip(*batches)
            emg=torch.stack(emg)
            glove=torch.stack(glove)
            labels=torch.stack(labels)
            batch=(emg,glove,labels,self.dt.domain)
            yield batch

    @property
    def amt_batches(self):
        if self.drop_last:
            return ( self.dt.D // self.batch_size )
        return math.ceil( self.dt.D / self.batch_size)

    def __len__(self):
        return self.amt_batches * self.dt.PEOPLE

class TaskWrapper():
    def __init__(self, dataset):
        self.dataset=dataset
        self.device=torch.device("cuda")
        #self.reset()
        
    def return_rand_g(self, D):
        b=torch.arange(self.dataset.TASKS, device=self.device, dtype=torch.long).reshape(self.dataset.TASKS,1)*D
        return torch.rand((self.dataset.TASKS, D), device=self.device).argsort(dim=-1)+b

    def return_rand(self, D):
        b=torch.arange(self.dataset.TASKS, device=self.device, dtype=torch.long).reshape(1,self.dataset.TASKS,1)*D
        return torch.rand((self.dataset.PEOPLE, self.dataset.TASKS, D), device=self.device).argsort(dim=-1)+b

    def reset(self):
        self.emg_rand=self.return_rand(self.dataset.D).to(torch.long)
        self.glove_rand=self.return_rand_g(self.dataset.glover.D).to(torch.long)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __len__(self):
        return self.dataset.D*self.dataset.PEOPLE

    def __getitem__(self, idx):
        people_idx=idx//self.dataset.D
        else_idx=idx%self.dataset.D
        tensor_emg = self.dataset.slice_batch(people_idx, self.emg_rand[people_idx, :, else_idx])
        tensor_glove = self.dataset.glover[self.glove_rand[:, idx%self.dataset.glover.D]]
        label=torch.arange(self.dataset.TASKS, device=self.device, dtype=torch.long)

        # move to half precision
        tensor_emg=tensor_emg.to(torch.float32)
        tensor_glove=tensor_glove.to(torch.float32)

        self.domain=self.dataset.domain
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
    def __init__(self, path_dir=None, complete=False, mode=False):
        self.counter = 0
        self.complete = complete
        self.path=path_dir
        self.mode=mode

    def push(self, X):
        self.counter+=1
        self.np = isinstance(X, np.ndarray)

        
        if self.mode:
            X=X.mean((1,2))
        else:
            X=X.mean(0)

        if self.counter==1:
            self.old_mean = self.new_mean = X
            self.old_std = np.zeros(X.shape, dtype=X.dtype) if self.np else torch.zeros(X.shape, device=X.device, dtype=X.dtype)
            self.ones = np.ones(X.shape, dtype=X.dtype) if self.np else torch.ones(X.shape, device=X.device, dtype=X.dtype)
        else:
            self.new_mean=self.old_mean + (X-self.old_mean)/self.counter
            self.new_std=self.old_std+(X-self.old_mean)*(X-self.new_mean)
            self.old_mean=self.new_mean
            self.old_std=self.new_std

    def mean(self):
        mean=self.new_mean
        if self.complete:
            mean=mean.mean()
        if self.path is not None:
            if not self.np:
                np.save(self.path+"mean.npy", mean.cpu().numpy())
            else:
                np.save(self.path+"mean.npy", mean)
        return mean

    def variance(self):
        if self.counter==1:
            return self.ones
        return (self.new_std)/(self.counter-1)

    def std(self):
        var = self.variance()
        if self.complete:
            var = var.mean()
        std=self.variance().sqrt()

        if self.path is not None:
            if self.np:
                np.save(self.path+"std.npy", std)
            else:
                np.save(self.path+"std.npy", std.cpu().numpy())
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
