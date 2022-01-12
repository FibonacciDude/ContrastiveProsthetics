#!/bin/python3

import torch
import torch.utils.data as data
import scipy.io as sio
from scipy import signal
from constants import *
import pandas as pd
import time
import numpy as np
import tqdm
from utils import RunningStats
from torch.multiprocessing import Pool, Process, set_start_method
import pyxis as px
try:
     set_start_method('spawn')
except RuntimeError:
    pass

torch.manual_seed(42)
#np.set_printoptions(precision=4, suppress=True)
#torch.set_printoptions(precision=4)

def get_np(dbnum, p_dir, n):
    E_mat=sio.loadmat("../db%s/s%s/S%s_E%s_A1"%(dbnum,p_dir,p_dir,n))
    emg=E_mat['emg'] # 12
    acc=E_mat['acc'] # 36
    glove=E_mat['glove'] # 22
    stimulus=E_mat['restimulus']
    repetition=E_mat['rerepetition']
    return emg, acc, glove, stimulus, repetition


class DB23(data.Dataset):
    def __init__(self, train=True, val=False, device="cuda"):
        self.device=torch.device(device)
        self.train=train
        self.val=val
        self.raw=False

        # if you randomize the tasks, you should randomize the 
        # labels...
        self.tasks_train=self.torchize(TRAIN_TASKS)
        self.tasks_test=self.torchize(TEST_TASKS)

        self.people_rand_train=torch.randperm(MAX_PEOPLE_TRAIN, device=self.device)
        self.people_rand=torch.randperm(MAX_PEOPLE, device=self.device)

        _rand_perm_train=torch.randperm(MAX_TRAIN_REPS, device=self.device)
        _rand_perm_test=torch.randperm(MAX_TEST_REPS, device=self.device)

        train_reps = self.torchize(TRAIN_REPS)
        test_reps = self.torchize(TEST_REPS)
        self.rep_rand_train=train_reps[_rand_perm_train[:-1]]-1
        self.rep_rand_val=train_reps[_rand_perm_train[-1:]]-1
        self.rep_rand_test=test_reps[_rand_perm_test]-1

        self.window_mask=torch.randperm(WINDOW_OUTPUT_DIM-1, device=self.device)

        self.path="/home/breezy/ULM/prosthetics/db23/"
        #self.path="../"

    def torchize(self, X):
        return torch.from_numpy(np.array(X)).to(self.device)

    def set_train(self):
        self.train=True
        self.val=False

    def set_test(self):
        self.train=False
        self.val=False

    def set_val(self):
        self.train=False
        self.val=True


    def __len__(self):
        if self.train:
            reps = (MAX_TRAIN_REPS-1)//BLOCK_SIZE
        elif self.val:
            reps = 1
        else:
            reps = MAX_TEST_REPS//BLOCK_SIZE
        return self.PEOPLE*reps*MAX_WINDOW_BLOCKS

    def size(self): # datapoints
        if self.train:
            dims=(MAX_PEOPLE_TRAIN,MAX_TASKS_TRAIN+1,MAX_TRAIN_REPS-1,WINDOW_STRIDE, MAX_WINDOW_BLOCKS)
            return np.prod(dims), dims
        elif self.val:
            dims=(MAX_PEOPLE_TRAIN,MAX_TASKS_TRAIN+1,1,WINDOW_STRIDE,MAX_WINDOW_BLOCKS)
            return np.prod(dims), dims
        else:
            dims=(MAX_PEOPLE,MAX_TASKS+1,MAX_TEST_REPS,WINDOW_STRIDE,MAX_WINDOW_BLOCKS)
            return np.prod(dims), dims

    def load_stored(self):
        self.EMG=torch.load(self.path+'data/emg.pt', map_location=self.device)
        self.ACC=torch.load(self.path+'data/acc.pt', map_location=self.device)
        self.GLOVE=torch.load(self.path+'data/glove.pt', map_location=self.device)
        self.EMG.cuda()
        self.ACC.cuda()
        self.GLOVE.cuda()

    def load_subject(self, subject):
        EMG=self.EMG[subject]
        ACC=self.ACC[subject]
        GLOVE=self.GLOVE[subject]
        return EMG,ACC,GLOVE

    def save(self, tensors):
        EMG, ACC, GLOVE = tensors
        torch.save(EMG, self.path+'data/emg.pt')
        torch.save(ACC, self.path+'data/acc.pt')
        torch.save(GLOVE, self.path+'data/glove.pt')

    # credit to github user parasgulati8
    def filter(self, data, f, butterworth_order=4,btype="bandpass"):
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
            # TODO: see effect
            transpose[i] = (signal.filtfilt(b, a, transpose[i]))
        return transpose.T

    def remove_outliers(self, tensor, dim, low, high, factor=1):
        for d in range(dim):
            tensor.T[d]=np.clip(tensor.T[d], a_min=low[d], a_max=high[d])
        return tensor

    def get_stim_rep(self, stimulus, repetition):
        # stimulus from 1-40, repetition from 1-6
        ex=np.searchsorted(np.array(TASK_DIST).cumsum(), stimulus)
        emg, acc, glove, stim, rep = self.Es[ex]
        # emg, acc, glove, stimulus, repetition
        stim_mask, rep_mask=(stim==stimulus), (rep==repetition)
        mask=(stim_mask&rep_mask).squeeze()

        # TODO: check effect of removing 11th sensor 
        # (this was removed in https://www.nature.com/articles/s41597-019-0349-2 for noise problems as well)
        glove=np.concatenate((glove[:, :10], glove[:, 11:]), axis=1)
        emg_=emg[mask][self.time_mask] * 10**4  # move to more dense region
        acc_=acc[mask][self.time_mask]
        glove_=glove[mask][self.time_mask]

        # filter - 10 Hz to 500 Hz
        emg_=self.filter(emg_, (20, 500), butterworth_order=4,btype="bandpass") # bandpass filter
        emg_=self.torchize(emg_)
        glove_=self.torchize(glove_)
        acc_=self.torchize(acc_)
        return emg_, acc_, glove_

    def clip_es(self, Es):
        Es_new = []
        for ex in range(2):
            emg, acc, glove, stim, rep = Es[ex]
            eh,el=np.percentile(emg, [99, 1], axis=0)
            ah,al=np.percentile(acc, [99, 1], axis=0)
            gh,gl=np.percentile(glove, [99, 1], axis=0)
            glove=self.remove_outliers(glove, GLOVE_DIM, low=gl, high=gh, factor=1)
            acc=self.remove_outliers(acc, ACC_DIM, low=al, high=ah, factor=1)
            emg=self.remove_outliers(emg, EMG_DIM, low=el, high=eh, factor=1)
            Es_new.append((emg, acc, glove, stim, rep))
        return tuple(Es_new)

    def load_dataset(self):
        """
        Loads dataset as a pt file format all preprocessed.
        subject -> reps -> stim -> amt windows -> window_ms (1 frame per ms) -> dim (emg,acc,glove)
        """
        PEOPLE=PEOPLE_D2+PEOPLE_D3
        print("People:", PEOPLE)
        self.time_mask=np.arange(10,TOTAL_WINDOW_SIZE*FACTOR+10,FACTOR,dtype=np.uint8)
        self.emg_stats=RunningStats()
        self.acc_stats=RunningStats()
        self.glove_stats=RunningStats()

        shape=(len(PEOPLE), MAX_TASKS+1,MAX_REPS,TOTAL_WINDOW_SIZE)
        EMG=torch.empty(shape+(EMG_DIM,), device=self.device)
        ACC=torch.empty(shape+(ACC_DIM,), device=self.device)
        GLOVE=torch.empty(shape+(GLOVE_DIM,), device=self.device)

        for i, person in enumerate(tqdm.tqdm(PEOPLE)):
            # gestures go from 1 to 17, 1 to 23, rest (0)
            # emg, acc, glove, stimulus, repetition
            self.person=person
            dbnum="3" if person >= MAX_PEOPLE_D2 else "2"
            subject=person
            if dbnum=="3":
                subject %= MAX_PEOPLE_D2
            p_dir=str(subject+1)

            # Fetch from .mat file
            E1=get_np(dbnum,p_dir,"1")
            E2=get_np(dbnum,p_dir,"2")
            self.Es = (E1, E2)
            # clip tensors of outliers
            self.Es=self.clip_es(self.Es)

            for rep in range(0,MAX_REPS):
                for stim in range(MAX_TASKS+1):
                    emg,acc,glove=self.get_stim_rep(stim,rep+1)
                    # only add if in the training set
                    if (person in TRAIN_PEOPLE and rep in self.rep_rand_train and stim in TRAIN_TASKS):
                        self.emg_stats.push(emg) 
                        self.acc_stats.push(acc) 
                        self.glove_stats.push(glove) 

                    EMG[i,stim,rep]=emg
                    ACC[i,stim,rep]=acc
                    GLOVE[i,stim,rep]=glove

        emg_means,emg_std=self.emg_stats.mean_std()
        acc_means,acc_std=self.acc_stats.mean_std()
        glove_means,glove_std=self.glove_stats.mean_std()
        print(emg_std,acc_std,glove_std)

        # normalize
        EMG=(EMG-emg_means)/emg_std
        ACC=(ACC-acc_means)/acc_std
        GLOVE=(GLOVE-glove_means)/glove_std
        #save
        self.save((EMG,ACC,GLOVE)) 
        print("Dataset (un)loading completed successfully")

    @property
    def PEOPLE(self):
        return MAX_PEOPLE_TRAIN if (self.train or self.val) else MAX_PEOPLE 

    @property
    def TASKS(self):
        return MAX_TASKS_TRAIN+1 if (self.train or self.val) else MAX_TASKS+1

    @property
    def tasks_mask(self):
        tasks_mask=self.tasks_train if (self.train or self.val) else self.tasks_test
        tasks_mask=torch.cat((tasks_mask, self.torchize([0])))
        return tasks_mask

    @property
    def people_mask(self):
        return self.people_rand_train if (self.train or self.val) else self.people_rand

    @property
    def block_mask(self):
        if self.train:
            return self.rep_rand_train
        elif self.val:
            return self.rep_rand_val
        else:
            return self.rep_rand_test

    def get_idx_(self, idx):
        mul = self.PEOPLE * MAX_WINDOW_BLOCKS
        rep_block = idx // mul
        sub_wind_idx = idx % mul
        subject = sub_wind_idx // MAX_WINDOW_BLOCKS
        window_block = sub_wind_idx % MAX_WINDOW_BLOCKS
        return (rep_block, subject, window_block)

    def slice_batch(self, tensor, tmask, bmask, wmask, dim):
        shape=(BLOCK_SIZE*self.TASKS, TOTAL_WINDOW_SIZE, dim)
        tensor=tensor[tmask, bmask].reshape(shape)
        if self.raw:
            return tensor
        stride = (TOTAL_WINDOW_SIZE*dim, WINDOW_STRIDE*dim, dim, 1)
        # ahh, subtle. The last output value would not be complete as STRIDE < WINDOW_MS
        # so it can only be WINDOW_MS size
        tensor=torch.as_strided(tensor, (BLOCK_SIZE*self.TASKS, WINDOW_OUTPUT_DIM-1, WINDOW_MS, dim), stride)
        # take random sample from this window
        # shape (TASKS*BLOCK_SIZE,WINDOW_BLOCK,WINDOW_MS,DIM)
        # (TASKS*BLOCK_SIZE*WINDOW_BLOCK,WINDOW_MS,DIM)
        #print(tensor[:, wmask].shape, self.train and not self.val)
        return tensor[:, wmask].reshape(-1, 1, WINDOW_MS, dim)


    def __getitem__(self, batch_idx):
        # batch_idx -> rep_block x (subject x window_block)
        # return batch of shape ( rep * amtwindows * maxtask) x windowms x respective dim
        # acc dimension is just a mean
        # glove and emg are images (windowms x respective dim)
        # Everything is randomized. No need to shuffle dataloader.
        rep_block, subject, window_block = self.get_idx_(batch_idx)

        block_mask=self.block_mask[rep_block*BLOCK_SIZE:(rep_block+1)*BLOCK_SIZE]
        window_mask=self.window_mask[window_block*WINDOW_BLOCK:(window_block+1)*WINDOW_BLOCK]
        tasks_mask=self.tasks_mask

        subject = self.people_mask[subject]
        # shape = (41, 6, window_ms, amt_windows, dim), specific case
        EMG,ACC,GLOVE=self.load_subject(subject)
    
        EMG=self.slice_batch(EMG, tasks_mask, block_mask, window_mask, EMG_DIM)
        GLOVE=self.slice_batch(GLOVE, tasks_mask, block_mask, window_mask, GLOVE_DIM)
        ACC=self.slice_batch(ACC, tasks_mask, block_mask, window_mask, ACC_DIM)
        ACC=ACC.squeeze(1).mean(1)
        #print(EMG.shape)
        return (EMG,GLOVE,ACC)


def load(db):
    db.load_dataset()

def info(db):
    print("New tasks", db.task_rand[-NEW_TASKS:].cpu().numpy())
    for train in [False, True]:
        if train:
            db.set_train()
            e,g,a=db[0]
            print(e.shape)
            print(g.min(), g.max(), g.mean(), g.std())
        else:
            db.set_test()
            e,g,a=db[0]
            print(e.shape)
            print(g.min(), g.max(), g.mean(), g.std())
        print("TRAIN:" if train else "TEST:")
        size,size_dims=db.size()
        print("\tDatapoints: dim (%s), size %s"%(size,size_dims))
        batch=len(db)
        print("\tBatch amts: %s"%(batch))
        
def visualize(db):
    import matplotlib.pyplot as plt
    db.raw=True
    EMG,GLOVE,ACC=db[0]
    #EMG=EMG.cpu().numpy()
    #for sensor in range(EMG_DIM):
    #    plt.plot(EMG[0, :, sensor])
    GLOVE=GLOVE.cpu().numpy()
    for sensor in range(GLOVE_DIM):
        plt.plot(GLOVE[10, :, sensor])
    plt.show()
    
if __name__=="__main__":
    db=DB23(device="cpu")
    #load(db)
    db.load_stored()
    info(db)
    #t=time.time()
    #for i in range(100):
    #    a=db[i]
    #print((time.time()-t)/100)
    #visualize(db)
