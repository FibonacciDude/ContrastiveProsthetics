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
np.set_printoptions(precision=4, suppress=True)
torch.set_printoptions(precision=4)

def get_np(dbnum, p_dir, n):
    E_mat=sio.loadmat("../db%s/s%s/S%s_E%s_A1"%(dbnum,p_dir,p_dir,n))
    emg=E_mat['emg'] # 12
    acc=E_mat['acc'] # 36
    glove=E_mat['glove'] # 22
    stimulus=E_mat['restimulus']
    repetition=E_mat['rerepetition']
    return emg, acc, glove, stimulus, repetition


def torchize(X, device="cuda"):
    device=torch.device(device)
    return torch.tensor(X, device=device)

class DB23(data.Dataset):
    def __init__(self, new_people=2, new_tasks=4, train=True, device="cuda"):
        self.device=torch.device(device)
        self.train=train

        self.NEW_PEOPLE=new_people
        self.NEW_TASKS=new_tasks
        self.MAX_PEOPLE_TRAIN=MAX_PEOPLE-new_people
        self.MAX_TASK_TRAIN=MAX_TASKS-self.NEW_TASKS

        # DO NOT take last *new_people*
        self.people_rand_d2=torch.randperm(MAX_PEOPLE_D2, device=self.device)
        self.people_rand_d3=torch.randperm(MAX_PEOPLE_D3, device=self.device)
        # Take by blocks of *2*
        self.rep_rand_train=torchize(TRAIN_REPS)[torch.randperm(MAX_TRAIN_REPS)]
        self.rep_rand_test=torchize(TEST_REPS)[torch.randperm(MAX_TEST_REPS)]

        self.path="/home/breezy/ULM/prosthetics/db23/"
        #self.path="../"

    def __len__(self):
        if self.train:
            length=self.MAX_PEOPLE_TRAIN*(MAX_TASKS-self.NEW_TASKS)*MAX_TRAIN_REPS
        else:
            length=self.MAX_PEOPLE_TRAIN*(MAX_TASKS-self.NEW_TASKS)*MAX_TEST_REPS+(MAX_PEOPLE)*MAX_TASKS*MAX_REPS
        return length * AMT_WINDOWS # amt windows per task in 150 ms total window

    def load_subject(self, subject):
        global MAX_PEOPLE_D2
        dbnum="3" if subject >= MAX_PEOPLE_D2 else "2"
        if dbnum=="3":
            subject %= MAX_PEOPLE_D2
        p_dir=str(subject+1)
        #load and normalize
        EMG=torch.load(self.path+'db%s/s%s/emg.pt'%(dbnum,p_dir))
        ACC=torch.load(self.path+'db%s/s%s/acc.pt'%(dbnum,p_dir))
        GLOVE=torch.load(self.path+'db%s/s%s/glove.pt'%(dbnum,p_dir))
        return EMG,ACC,GLOVE

    def save_subject(self, subject, tensors):
        EMG, ACC, GLOVE = tensors
        global MAX_PEOPLE_D2
        dbnum="3" if subject >= MAX_PEOPLE_D2 else "2"
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
        if mask.sum() < TOTAL_WINDOW_SIZE*(Hz//1000):
            print("Smaller size: %s, stimulus %s, repetition %s"%(str(self.person), str(stimulus), str(repetition)))
            time_mask=np.array([np.arange(0,Hz//1000*WINDOW_MS,Hz//1000,dtype=np.uint8)]*AMT_WINDOWS)
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

        # filter
        emg_=self.filter(emg_, (20, 40), butterworth_order=4,btype="bandpass") # bandpass filter

        # take out outliers
        iqr_glove=np.subtract(*np.percentile(glove_, [75, 25], axis=0))
        iqr_acc=np.subtract(*np.percentile(acc_, [75, 25], axis=0))
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

        acc_=acc_.reshape(AMT_WINDOWS, -1, ACC_DIM).mean(1)

        return emg_, acc_, glove_


    def load_dataset(self, norm=False):
        """
        Loads dataset as a pt file format all preprocessed.
        subject -> reps -> stim -> amt windows -> window_ms (1 frame per ms) -> dim (emg,acc,glove)
        """
        PEOPLE=PEOPLE_D2+PEOPLE_D3
        self.time_mask=np.arange(0,Hz//1000*TOTAL_WINDOW_SIZE,Hz//1000,dtype=np.uint8)

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
                ACC=torch.empty(shape[:-1]+(AMT_WINDOWS, ACC_DIM,),device=self.device)
                GLOVE=torch.empty(shape+(GLOVE_DIM,),device=self.device)

                for rep in range(1,MAX_REPS+1):
                    for stim in range(MAX_TASKS+1):
                        emg,acc,glove=self.get_stim_rep(stim,rep)
                        emg,acc,glove=torchize(emg),torchize(acc),torchize(glove)

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

    def __getitem__(self, batch_idx):
        # batch_idx -> rep_block x subject
        # return batch of shape (2 (rep) * amtwindows * maxtask) x windowms x respective dim
        # acc dimension is just a mean
        # glove and emg are images (windowms x respective dim)

        BLOCK_SIZE=2
        PEOPLE=MAX_PEOPLE_TRAIN if self.train else MAX_PEOPLE
        TASKS=self.MAX_TASKS_TRAIN if self.train else MAX_TASKS

        rep_block = batch_idx // PEOPLE
        subject = batch_idx % PEOPLE 
        

        return self.load_subject(batch_idx)

if __name__=="__main__":
    t=time.time()
    db=DB23()
    db.load_dataset()
    print(time.time()-t)

    #t=time.time()
    #for i in PEOPLE_D3+PEOPLE_D2:
    #    EMG,ACC,GLOVE=db[i]
    #    print(EMG.shape, i)
    #print(time.time()-t)
