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


def torchize(X, device="cuda"):
    if isinstance(device, str):
        device=torch.device(device)
    return torch.tensor(X, device=device)

class DB23(data.Dataset):
    def __init__(self, new_people=2, new_tasks=4, train=True, val=False,device="cuda"):
        self.device=torch.device(device)
        self.train=train
        self.val=val

        self.NEW_PEOPLE=new_people
        self.NEW_TASKS=new_tasks
        self.MAX_PEOPLE_TRAIN=MAX_PEOPLE-new_people
        self.MAX_TASKS_TRAIN=MAX_TASKS-self.NEW_TASKS

        # DO NOT take last *new_people*
        self.people_rand_d2=torch.randperm(MAX_PEOPLE_D2, device=self.device)
        self.people_rand_d3=torch.randperm(MAX_PEOPLE_D3, device=self.device)
        # Take by blocks of *2*
        _rand_perm_test=torch.randperm(MAX_TEST_REPS, device=self.device)
        _rand_perm_train=torch.randperm(MAX_TRAIN_REPS, device=self.device)

        train_reps = torchize(TRAIN_REPS, device=device)
        test_reps = torchize(TEST_REPS, device=device)
        self.rep_rand_train=train_reps[_rand_perm_train[:-1]]-1
        self.rep_rand_val=train_reps[_rand_perm_train[-1:]]-1
        self.rep_rand_test=test_reps[_rand_perm_test]-1
        self.task_rand=torch.randperm(MAX_TASKS, device=self.device)

        # for voting later?
        #self.window=torch.arange(end=WINDOW_OUTPUT_DIM, device=self.device)
        self.window=torch.randperm(WINDOW_OUTPUT_DIM, device=self.device)

        self.path="/home/breezy/ULM/prosthetics/db23/"
        #self.path="../"

    def __len__(self):
        if self.train:
            return self.MAX_PEOPLE_TRAIN*((MAX_TRAIN_REPS-1)//BLOCK_SIZE)*MAX_WINDOW_BLOCKS
        elif self.val:
            return self.MAX_PEOPLE_TRAIN*1*MAX_WINDOW_BLOCKS
        else:
            # TODO: Change
            return MAX_PEOPLE*(MAX_TEST_REPS//BLOCK_SIZE)*MAX_WINDOW_BLOCKS

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
            #dims=(MAX_PEOPLE,MAX_TASKS,MAX_TEST_REPS,WINDOW_STRIDE,MAX_WINDOW_BLOCKS)
            dims=(MAX_PEOPLE,self.MAX_TASKS_TRAIN,MAX_TEST_REPS,WINDOW_STRIDE,MAX_WINDOW_BLOCKS)
            return np.prod(dims), dims

    def load_subject(self, subject):
        global MAX_PEOPLE_D2
        dbnum="3" if (subject >= MAX_PEOPLE_D2) else "2"
        if dbnum=="3":
            subject %= MAX_PEOPLE_D2
        p_dir=str(subject+1)
        #load and normalize
        EMG=torch.load(self.path+'db%s/s%s/emg.pt'%(dbnum,p_dir))
        ACC=torch.load(self.path+'db%s/s%s/acc.pt'%(dbnum,p_dir))
        GLOVE=torch.load(self.path+'db%s/s%s/glove.pt'%(dbnum,p_dir))
        EMG.to(self.device)
        ACC.to(self.device)
        GLOVE.to(self.device)
        return EMG,ACC,GLOVE

    def save_subject(self, subject, tensors):
        EMG, ACC, GLOVE = tensors
        global MAX_PEOPLE_D2
        dbnum="3" if (subject >= MAX_PEOPLE_D2) else "2"
        if dbnum=="3":
            subject %= MAX_PEOPLE_D2
        p_dir=str(subject+1)
        #save
        torch.save(EMG, self.path+'db%s/s%s/emg.pt'%(dbnum,p_dir))
        torch.save(ACC, self.path+'db%s/s%s/acc.pt'%(dbnum,p_dir))
        torch.save(GLOVE, self.path+'db%s/s%s/glove.pt'%(dbnum,p_dir))

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
        transpose = data.T#.copy()
        
        for i in range(len(transpose)):
            transpose[i] = (signal.lfilter(b, a, transpose[i]))
        return transpose.T

    def get_stim_rep(self, stimulus, repetition): # stimulus from 1-40, repetition from 1-6
        global TASK_DIST
        TASK_DIST=np.array(TASK_DIST)
        ex=np.searchsorted(TASK_DIST.cumsum(), stimulus)
        emg, acc, glove, stim, rep = self.Es[ex]
        # emg, acc, glove, stimulus, repetition
        stim_mask, rep_mask=(stim==stimulus), (rep==repetition) #(0 if stimulus==0 else repetition))
        mask=(stim_mask&rep_mask).squeeze()
        if mask.sum() < TOTAL_WINDOW_SIZE:
            print("Smaller size: %s, stimulus %s, repetition %s, %s"%(str(self.person), str(stimulus), str(repetition), str(mask.sum())))
            # start 100 ms after it says it has started
            time_mask=np.array([np.arange(WINDOW_MS,dtype=np.uint8)]*AMT_WINDOWS)
            emg_=emg[mask][time_mask] * 16 # 15 ms window
            acc_=acc[mask][time_mask] * 16
            glove_=glove[mask][time_mask] * 16
            emg_=emg_.reshape(-1, EMG_DIM)
            acc_=acc_.reshape(-1, ACC_DIM)
            glove_=glove_.reshape(-1, GLOVE_DIM)
        else:
            emg_=emg[mask][self.time_mask] * 16  # bit shift to avoid having too low of a number
            acc_=acc[mask][self.time_mask] * 16
            glove_=glove[mask][self.time_mask] * 16 # / 2**3

        # filter - 5 Hz to 500 Hz
        emg_=self.filter(emg_, (10, 500), butterworth_order=4,btype="bandpass") # bandpass filter
        # take out outliers
        iqr_glove=np.subtract(*np.percentile(glove_, [75, 25], axis=0))
        iqr_acc=np.subtract(*np.percentile(acc_, [75, 25], axis=0))
        # take out anything below the 1 percentile or above 99 percentile
        emg_high,emg_low=np.percentile(acc_, [99, 1], axis=0)

        for dim in range(GLOVE_DIM):
            glove_.T[dim][(glove_.T[dim]>1.5*iqr_glove[dim])] = iqr_glove[dim]
            glove_.T[dim][(glove_.T[dim]<-1.5*iqr_glove[dim])] = -iqr_glove[dim]

        for dim in range(ACC_DIM):
            acc_.T[dim][(acc_.T[dim]>1.5*iqr_acc[dim])] = iqr_acc[dim]
            acc_.T[dim][(acc_.T[dim]<-1.5*iqr_acc[dim])] = -iqr_acc[dim]

        for dim in range(EMG_DIM):
            emg_.T[dim][(emg_.T[dim]>emg_high[dim])] = emg_high[dim]
            emg_.T[dim][(emg_.T[dim]<emg_low[dim])] = -emg_low[dim]

        #acc_=acc_.reshape(AMT_WINDOWS, -1, ACC_DIM).mean(1) # no mean anymore so we can stride

        return emg_, acc_, glove_


    def load_dataset(self, norm=False):
        """
        Loads dataset as a pt file format all preprocessed.
        subject -> reps -> stim -> amt windows -> window_ms (1 frame per ms) -> dim (emg,acc,glove)
        """
        PEOPLE=PEOPLE_D2+PEOPLE_D3
        self.time_mask=np.arange(TOTAL_WINDOW_SIZE,dtype=np.uint8)
        tasks_mask_train=self.task_rand[:-self.NEW_TASKS] # if (self.train or self.val) else self.task_rand

        self.emg_stats=RunningStats()
        self.acc_stats=RunningStats()
        self.glove_stats=RunningStats()

        for person in tqdm.tqdm(PEOPLE):
            # gestures go from 1 to 17, 1 to 23, rest (0)
            # emg, acc, glove, stimulus, repetition
            self.person=person
            dbnum="3" if person >= MAX_PEOPLE_D2 else "2"
            subject=person
            if dbnum=="3":
                subject %= MAX_PEOPLE_D2
            p_dir=str(subject+1)

            if not norm:
                # Fetch from .mat file
                E1=get_np(dbnum,p_dir,"1")
                E2=get_np(dbnum,p_dir,"2")
                self.Es = (E1, E2)

                shape=(MAX_TASKS+1,MAX_REPS,TOTAL_WINDOW_SIZE)
                EMG=torch.empty(shape+(EMG_DIM,),device=self.device)
                ACC=torch.empty(shape+(ACC_DIM,),device=self.device)
                GLOVE=torch.empty(shape+(GLOVE_DIM,),device=self.device)

                for rep in range(1,MAX_REPS+1):
                    for stim in range(MAX_TASKS+1):
                        emg,acc,glove=self.get_stim_rep(stim,rep)
                        emg,acc,glove=torchize(emg),torchize(acc),torchize(glove)

                        # only add if in the training set
                        if (person in self.people_rand_d2 or person in self.people_rand_d3[:-self.NEW_PEOPLE]) and rep in self.rep_rand_train and stim in tasks_mask_train:
                            self.emg_stats.push(emg) 
                            self.acc_stats.push(acc) 
                            self.glove_stats.push(glove) 

                        EMG[stim,rep-1]=emg
                        ACC[stim,rep-1]=acc
                        GLOVE[stim,rep-1]=glove

            else:
                # Fetch from .pt file
                EMG,ACC,GLOVE=self.load_subject(person)


            if not norm:
                EMG=EMG.reshape(EMG.shape[:-2]+(AMT_WINDOWS, WINDOW_MS, EMG_DIM))
                GLOVE=GLOVE.reshape(GLOVE.shape[:-2]+(AMT_WINDOWS, WINDOW_MS, GLOVE_DIM))
                #save
                self.save_subject(person, (EMG,ACC,GLOVE))

        emg_means,emg_std=self.emg_stats.mean_std()
        acc_means,acc_std=self.acc_stats.mean_std()
        glove_means,glove_std=self.glove_stats.mean_std()
        print(emg_std,acc_std,glove_std)

        # normalize
        for person in tqdm.tqdm(PEOPLE):
            
            EMG,ACC,GLOVE=self.load_subject(person)
            EMG=(EMG-emg_means)/emg_std
            ACC=(ACC-acc_means)/acc_std
            GLOVE=(GLOVE-glove_means)/glove_std
            #save
            self.save_subject(person, (EMG,ACC,GLOVE))

        print("Dataset (un)loading completed successfully")
        return None

    @property
    def PEOPLE(self):
        return self.MAX_PEOPLE_TRAIN if (self.train or self.val) else MAX_PEOPLE

    @property
    def TASKS(self):
        # TODO: Change this back
        return self.MAX_TASKS_TRAIN+1 # if (self.train or self.val) else MAX_TASKS+1

    @property
    def tasks_mask(self):
        #tasks_mask=self.task_rand[:-self.NEW_TASKS] if (self.train or self.val) else self.task_rand
        # TODO: remove, this. This is to test hypothesis that the new tasks are "messing up" the batch normalization
        tasks_mask=self.task_rand[:-self.NEW_TASKS] # if (self.train or self.val) else self.task_rand
        tasks_mask=torch.cat((tasks_mask, torchize([0],device=self.device)))
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

        rep_block, subject, window_block = self.get_idx_(batch_idx)

        block_mask=self.block_mask[rep_block*BLOCK_SIZE:(rep_block+1)*BLOCK_SIZE]
        window_mask=self.window[window_block*WINDOW_BLOCK:(window_block+1)*WINDOW_BLOCK]
        tasks_mask=self.tasks_mask

        if subject >= MAX_PEOPLE_D2:
            subject = self.people_rand_d3[subject%MAX_PEOPLE_D2].item()
        else:
            subject = self.people_rand_d2[subject].item()

        # shape = (41, 6, window_ms, amt_windows, dim), specific case
        EMG,ACC,GLOVE=self.load_subject(subject)

        EMG=self.slice_batch(EMG, tasks_mask, block_mask, window_mask, EMG_DIM)
        GLOVE=self.slice_batch(GLOVE, tasks_mask, block_mask, window_mask, GLOVE_DIM)
        ACC=self.slice_batch(ACC, tasks_mask, block_mask, window_mask, ACC_DIM)
        ACC=ACC.squeeze(1).mean(1)

        return (EMG,GLOVE,ACC)


def load(db):
    db.load_dataset()

def info(db):
    for train in [False, True]:
        if train:
            db.set_train()
        else:
            db.set_test()
        print("TRAIN:" if train else "TEST:")
        size,size_dims=db.size()
        print("\tDatapoints: dim (%s), size %s"%(size,size_dims))
        batch=len(db)
        print("\tBatch amts: %s"%(batch))
        
if __name__=="__main__":
    db=DB23(new_people=3,new_tasks=4)
    load(db)
    #info(db)
