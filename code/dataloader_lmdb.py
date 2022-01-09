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
    def __init__(self, new_people=2, new_tasks=4, train=True, val=False, device="cuda"):
        self.device=torch.device(device)
        self.train=train
        self.val=val
        self.raw=False

        self.NEW_PEOPLE=new_people
        self.NEW_TASKS=new_tasks
        self.MAX_PEOPLE_TRAIN=MAX_PEOPLE-new_people
        self.MAX_TASKS_TRAIN=MAX_TASKS-self.NEW_TASKS

        self.task_rand=torch.randperm(MAX_TASKS, device=self.device)

        # DO NOT take last *new_people*
        self.people_rand_d2=torch.randperm(MAX_PEOPLE_D2, device=self.device)
        self.people_rand_d3=torch.randperm(MAX_PEOPLE_D3, device=self.device)
        # Take by blocks of *2*
        _rand_perm_train=torch.randperm(MAX_TRAIN_REPS, device=self.device)
        _rand_perm_test=torch.randperm(MAX_TEST_REPS, device=self.device)

        train_reps = self.torchize(TRAIN_REPS)
        test_reps = self.torchize(TEST_REPS)
        self.rep_rand_train=train_reps[_rand_perm_train[:-1]]-1
        self.rep_rand_val=train_reps[_rand_perm_train[-1:]]-1
        self.rep_rand_test=test_reps[_rand_perm_test]-1


        #self.window=torch.arange(end=WINDOW_OUTPUT_DIM, device=self.device)
        self.window=torch.randperm(WINDOW_OUTPUT_DIM, device=self.device)

        self.path="/home/breezy/ULM/prosthetics/db23/"
        #self.path="../"

    def torchize(self, X):
        return torch.from_numpy(np.array(X)).to(self.device)

    def __len__(self):
        if self.train:
            return self.MAX_PEOPLE_TRAIN*((MAX_TRAIN_REPS-1)//BLOCK_SIZE)*MAX_WINDOW_BLOCKS
        elif self.val:
            return self.MAX_PEOPLE_TRAIN*1*MAX_WINDOW_BLOCKS
        else:
            # TODO: Change back
            return MAX_PEOPLE*(MAX_TEST_REPS//BLOCK_SIZE)*MAX_WINDOW_BLOCKS
            #return self.MAX_PEOPLE_TRAIN*(MAX_TEST_REPS//BLOCK_SIZE)*MAX_WINDOW_BLOCKS

    def set_train(self):
        self.train=True
        self.val=False

    def set_test(self):
        self.train=False
        self.val=False

    def set_val(self):
        self.train=False
        self.val=True

    def size(self): # datapoints
        if self.train:
            dims=(self.MAX_PEOPLE_TRAIN,self.MAX_TASKS_TRAIN,MAX_TRAIN_REPS-1,WINDOW_STRIDE, MAX_WINDOW_BLOCKS)
            return np.prod(dims), dims
        elif self.val:
            dims=(self.MAX_PEOPLE_TRAIN,self.MAX_TASKS_TRAIN,1,WINDOW_STRIDE,MAX_WINDOW_BLOCKS)
            return np.prod(dims), dims
        else:
            # TODO: Change back
            dims=(MAX_PEOPLE,MAX_TASKS,MAX_TEST_REPS,WINDOW_STRIDE,MAX_WINDOW_BLOCKS)
            #dims=(self.MAX_PEOPLE_TRAIN,self.MAX_TASKS_TRAIN,MAX_TEST_REPS,WINDOW_STRIDE,MAX_WINDOW_BLOCKS)
            return np.prod(dims), dims


    def load_db(self):
        self.db=px.Reader(dirpath=self.path+"data")

    def load_subject(self, subject):
        #load and normalize
        dat=self.db[subject]
        EMG,ACC,GLOVE=dat['emg'], dat['acc'], dat['glove']
        return EMG,ACC,GLOVE

    def save(self, tensors):
        EMG,ACC,GLOVE=tensors
        with px.Writer(dirpath=self.path+"data", map_size_limit=100_000, ram_gb_limit=20) as db:
            for s in range(MAX_PEOPLE):
                db.put_samples('emg', EMG, 'acc', ACC, 'glove', GLOVE)

    # credit to github user parasgulati8
    def filter(self, data, f, butterworth_order=4,btype="lowpass"):
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
        glove=np.concatenate((glove[:, :10], glove[:, 11:]), axis=1)
        emg_=emg[mask][self.time_mask] * 10**4  # move to more dense region
        acc_=acc[mask][self.time_mask]
        glove_=glove[mask][self.time_mask]

        # filter - 10 Hz to 500 Hz
        emg_=self.filter(emg_, (10, 500), butterworth_order=4,btype="bandpass") # bandpass filter
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
        print(PEOPLE)
        self.time_mask=np.arange(10,TOTAL_WINDOW_SIZE*FACTOR+10,FACTOR,dtype=np.uint8)
        tasks_mask_train=self.task_rand[:-self.NEW_TASKS] # if (self.train or self.val) else self.task_rand
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
                    if (person in self.people_rand_d2 or person in self.people_rand_d3[:-self.NEW_PEOPLE]) and rep in self.rep_rand_train and stim in tasks_mask_train:
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
        # TODO: Change this back
        return self.MAX_PEOPLE_TRAIN if (self.train or self.val) else MAX_PEOPLE

    @property
    def TASKS(self):
        # TODO: Change this back
        return self.MAX_TASKS_TRAIN+1 if (self.train or self.val) else MAX_TASKS+1

    @property
    def tasks_mask(self):
        #tasks_mask=self.task_rand[:-self.NEW_TASKS] if (self.train or self.val) else self.task_rand
        # TODO: remove, this. This is to test hypothesis that the new tasks are "messing up" the batch normalization
        tasks_mask=self.task_rand[:-self.NEW_TASKS] if (self.train or self.val) else self.task_rand
        tasks_mask=torch.cat((tasks_mask, self.torchize([0])))
        return tasks_mask

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
        # stride across the entire window (moving window of window_ms size)
        stride=(TOTAL_WINDOW_SIZE*dim, WINDOW_STRIDE, dim, 1)
        tensor=torch.as_strided(tensor, (shape[0], WINDOW_OUTPUT_DIM, WINDOW_MS, dim), stride)
        # take random sample from this window
        # shape (TASKS*BLOCK_SIZE,WINDOW_BLOCK,WINDOW_MS,DIM)
        # (TASKS*BLOCK_SIZE*WINDOW_BLOCK,WINDOW_MS,DIM)
        return tensor[:, wmask].reshape(-1, 1, WINDOW_MS, dim)


    def __getitem__(self, batch_idx):
        # batch_idx -> rep_block x (subject x window_block)
        # return batch of shape (2 (rep) * amtwindows * maxtask) x windowms x respective dim
        # acc dimension is just a mean
        # glove and emg are images (windowms x respective dim)
        t = time.time()
        rep_block, subject, window_block = self.get_idx_(batch_idx)

        block_mask=self.block_mask[rep_block*BLOCK_SIZE:(rep_block+1)*BLOCK_SIZE]
        window_mask=self.window[window_block*WINDOW_BLOCK:(window_block+1)*WINDOW_BLOCK]
        tasks_mask=self.tasks_mask

        if subject >= MAX_PEOPLE_D2:
            subject = self.people_rand_d3[subject%MAX_PEOPLE_D2].item()
        else:
            subject = self.people_rand_d2[subject].item()

        # shape = (41, 6, window_ms, amt_windows, dim), specific case
        t=time.time()
        EMG,ACC,GLOVE=self.load_subject(subject)
        print(time.time()-t, "time")

        EMG=self.slice_batch(EMG, tasks_mask, block_mask, window_mask, EMG_DIM)
        GLOVE=self.slice_batch(GLOVE, tasks_mask, block_mask, window_mask, GLOVE_DIM)
        ACC=self.slice_batch(ACC, tasks_mask, block_mask, window_mask, ACC_DIM)
        ACC=ACC.squeeze(1).mean(1)
        return (EMG,GLOVE,ACC)


def load(db):
    db.load_dataset()

def info(db):
    print("New tasks", db.task_rand[-db.NEW_TASKS:].cpu().numpy())
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
    print(EMG.shape)
    EMG=EMG.cpu().numpy()
    print(EMG.min(), EMG.max())
    for sensor in range(EMG_DIM):
         plt.plot(EMG[0, :, sensor])
    plt.show()
    
if __name__=="__main__":
    db=DB23(new_people=3,new_tasks=4, device="cpu")
    #load(db)
    #info(db)
    db.load_db()
    for i in range(100):
        a=db[0]
    visualize(db)
