#!/bin/python3

import torch
import torch.utils.data as data
import scipy.io as sio
from constants import *
import pandas as pd
import time
import numpy as np
import tqdm

torch.manual_seed(42)

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

    def get_stim_rep(self, stimulus, repetition): # stimulus from 1-40, repetition from 1-6
        global TASK_DIST
        TASK_DIST=np.array(TASK_DIST)
        ex=np.searchsorted(TASK_DIST.cumsum(), stimulus)
        emg, acc, glove, stim, rep = self.Es[ex]

        # emg, acc, glove, stimulus, repetition
        stim_mask, rep_mask=(stim==stimulus), (rep==(0 if stimulus==0 else repetition))
        mask=(stim_mask&rep_mask).squeeze()
        #return mask,ex
        emg_=emg[mask][self.time_mask]
        acc_=acc[mask][self.time_mask]
        acc_=acc_.reshape(AMT_WINDOWS, -1, ACC_DIM).mean(1)
        glove_=glove[mask][self.time_mask]
        return emg_, acc_, glove_

    def load_subject(self, subject):
        global MAX_PEOPLE_D2
        dbnum="3" if subject >= MAX_PEOPLE_D2 else "2"
        if dbnum=="2":
            person %= MAX_PEOPLE_D2
        p_dir=str(subject+1)
        #load and normalize
        EMG=torch.load(self.path+'db%s/s%s/emg.pt'%(dbnum,p_dir))
        ACC=torch.load(self.path+'db%s/s%s/acc.pt'%(dbnum,p_dir))
        GLOVE=torch.load(self.path+'db%s/s%s/glove.pt'%(dbnum,p_dir))
        return EMG,ACC,GLOVE

    def save(self, subject, tensors):
        EMG, ACC, GLOVE = tensors
        global MAX_PEOPLE_D2
        dbnum="3" if subject >= MAX_PEOPLE_D2 else "2"
        if dbnum=="2":
            subject %= MAX_PEOPLE_D2
        p_dir=str(subject+1)
        #save
        torch.save(EMG, self.path+'db%s/s%s/emg.pt'%(dbnum,p_dir))
        torch.save(ACC, self.path+'db%s/s%s/acc.pt'%(dbnum,p_dir))
        torch.save(GLOVE, self.path+'db%s/s%s/glove.pt'%(dbnum,p_dir))

    def load_dataset(self):
        """
        Loads dataset as a pt file format all preprocessed.
        subject -> reps -> amt windows -> window_ms (1 frame per ms) -> dim (emg,acc,glove)
        """
        #PEOPLE=PEOPLE_D2+PEOPLE_D3
        PEOPLE=[0,1]
        self.time_mask=np.arange(0,Hz//1000*TOTAL_WINDOW_SIZE,Hz//1000,dtype=np.uint8)

        emg_means,acc_means,glove_means=[],[],[]
        emg_sq,acc_sq,glove_sq=[],[],[]
        subject_totals=[]

        for person in tqdm.tqdm(PEOPLE):
            dbnum="3" if person >= MAX_PEOPLE_D2 else "2"
            if dbnum=="2":
                person %= MAX_PEOPLE_D2
            p_dir=str(person+1)

            # gestures go from 1 to 17, 1 to 23, 1 to 9, rest (0)
            # emg, acc, glove, stimulus, repetition
            E1=get_np(dbnum,p_dir,"1")
            E2=get_np(dbnum,p_dir,"2")
            self.Es = (E1, E2)
            subject_totals.append(len(E1[-1])+len(E2[-1]))

            #huge_mask, huge_enum=[],[]
            #for stimulus in 
            shape=(MAX_TASKS+1,MAX_REPS,TOTAL_WINDOW_SIZE)
            EMG=torch.empty(shape+(EMG_DIM,),device=self.device)
            ACC=torch.empty(shape[:-1]+(AMT_WINDOWS, ACC_DIM,),device=self.device)
            GLOVE=torch.empty(shape+(GLOVE_DIM,),device=self.device)

            emg_means.append(EMG.mean((0,1,2)))
            acc_means.append(ACC.mean((0,1,2)))
            glove_means.append(GLOVE.mean((0,1,2)))

            emg_sq.append((EMG**2).mean((0,1,2)))
            acc_sq.append((ACC**2).mean((0,1,2)))
            glove_sq.append((GLOVE**2).mean((0,1,2)))

            for rep in range(1,MAX_REPS+1):
                for stim in range(MAX_TASKS+1):
                    emg,acc,glove=self.get_stim_rep(stim,rep)
                    EMG[stim,rep-1]=torchize(emg)
                    ACC[stim,rep-1]=torchize(acc)
                    GLOVE[stim,rep-1]=torchize(glove)

            EMG=EMG.reshape(EMG.shape[:-2]+(AMT_WINDOWS, WINDOW_MS, EMG_DIM))
            GLOVE=GLOVE.reshape(GLOVE.shape[:-2]+(AMT_WINDOWS, WINDOW_MS, GLOVE_DIM))

            #save
            self.save(person, (EMG,ACC,GLOVE))

        sum_total=sum(subject_totals)
        subject_totals=torchize(subject_totals) / sum_total

        emg_means=torchize(emg_means * subject_totals).mean(0)
        acc_means=torchize(acc_means * subject_totals).mean(0)
        glove_means=torchize(glove_means * subject_totals).mean(0)

        emg_sq=torchize(emg_sq * subject_totals).mean(0)
        acc_sq=torchize(acc_sq * subject_totals).mean(0)
        glove_sq=torchize(glove_sq * subject_totals).mean(0)

        subject_totals *= sum_total
        emg_std=torchize(torch.sqrt(  (emg_means*sum_total) - (emg_sq/sum_total)  ))
        acc_std=torchize(torch.sqrt(  (acc_means*sum_total) - (acc_sq/sum_total)  ))
        glove_std=torchize(torch.sqrt(  (glove_means*sum_total) - (glove_sq/sum_total)  ))

        # normalize
        for person in tqdm.tqdm(PEOPLE):
            
            EMG,ACC,GLOVE=self.load(person)
            EMG=(EMG+emg_means)/emg_std
            ACC=(ACC+acc_means)/acc_std
            GLOVE=(GLOVE+glove_means)/glove_std
            #save
            self.save(person, (EMG,ACC,GLOVE))

        print("Dataset (un)loading completed successfully")

    def __getitem__(self, batch_idx):
        # batch of constant size of tasks*2 (train time)
        pass

db=DB23()
t=time.time()
print(len(db))
db.load_dataset()
print(time.time()-t)
