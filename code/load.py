#!/bin/python3
import argparse
import torch
import torch.utils.data as data
from constants import *
import pandas as pd
import time
import numpy as np
import tqdm
from utils import *
from torch.multiprocessing import Pool, Process, set_start_method
import pyxis as px
from utils import *
try:
     set_start_method('spawn')
except RuntimeError:
    pass

torch.manual_seed(42)

class DB23(data.Dataset):
    def __init__(self, train=True, val=False):
        self.device=torch.device("cuda")
        self.train=train
        self.val=val
        self.raw=False

        self.tasks_train=torchize(TRAIN_TASKS)
        self.tasks_test=torchize(TEST_TASKS)
        self.tasks=torchize(TASKS)

        self.people_train=torchize(TRAIN_PEOPLE_IDXS)
        self.people_test=torchize(TEST_PEOPLE_IDXS)
        self.people=torchize(PEOPLE_IDXS)

        train_reps = torchize(TRAIN_REPS)
        test_reps = torchize(TEST_REPS)
        reps = torchize(REPS)

        self.rep_train=train_reps[:-1]-1
        self.rep_val=train_reps[-1:]-1
        self.rep_test=test_reps-1
        self.reps=reps

        self.window_mask=torch.randperm(WINDOW_OUTPUT_DIM, device=self.device)
        self.path="/home/breezy/hci/prosthetics/db23/"

    def set_train(self):
        self.train=True
        self.val=False

    def set_test(self):
        self.train=False
        self.val=False

    def set_val(self):
        self.train=False
        self.val=True

    def load_stored(self):
        self.EMG=torch.load(self.path+'data/emg.pt', map_location=self.device)
        #self.EMG=self.EMG.transpose(0,1)
        self.EMG.cuda()

    def load_subject(self, subject):
        EMG=self.EMG[subject]
        return EMG

    def save(self, EMG):
        torch.save(EMG, self.path+'data/emg.pt')

    def get_stim_rep(self, stimulus, repetition):
        # stimulus from 1-40, repetition from 1-6
        ex=np.searchsorted(np.array(TASK_DIST).cumsum(), stimulus)
        emg, stim, rep = self.Es[ex]
        stim_mask, rep_mask=(stim==stimulus), (rep==repetition)
        mask=(stim_mask&rep_mask).squeeze()

        # this was removed in https://www.nature.com/articles/s41597-019-0349-2 for noise problems as well
        emg_=emg[mask][:TOTAL_WINDOW_SIZE+2*WINDOW_EDGE]
        # align data with emg

        # filter raw signal - 20 Hz to 400 Hz
        emg_=filter(emg_*2**10, (20, 400), butterworth_order=4,btype="bandpass") # bandpass filter
        # rectify signal - window of ~147
        emg_=rms(emg_)

        emg_=torchize(emg_[self.time_mask])
        return emg_

    def load_dataset(self):
        """
        Loads dataset as a pt file format all preprocessed.
        subject -> reps -> stim -> amt windows -> window_ms -> dim (emg)
        """
        global args
        print("People:", PEOPLE)
        self.time_mask=np.arange(0,TOTAL_WINDOW_SIZE,FACTOR,dtype=np.uint8)
        
        self.emg_stats=RunningStats(norm=args.norm, complete=False)

        shape=(len(PEOPLE), MAX_TASKS+1,MAX_REPS,len(self.time_mask))
        EMG=torch.empty(shape+(EMG_DIM,), device=self.device)

        for i, person in enumerate(tqdm.tqdm(PEOPLE)):
            # gestures go from 1 to 17, 1 to 23, rest (0)
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
            # clip tensors of outliers - no clipping for now
            #self.Es=self.clip_es(self.Es)

            for rep in range(0,MAX_REPS):
                for stim in range(MAX_TASKS+1):
                    emg=self.get_stim_rep(stim,rep+1)
                    # only add if in the training set
                    if (person in TRAIN_PEOPLE and rep in self.rep_train and (stim in TRAIN_TASKS or stim==0)):
                        self.emg_stats.push(emg) 

                    EMG[i,stim,rep]=emg

        if args.norm:
            emg_means,emg_std=self.emg_stats.mean_std()
            print(emg_std)
        else:
            print("Min max normalization:\n")
            print("\tEMG min max:", self.emg_stats.min.item(), self.emg_stats.max.item())

        # normalize
        EMG=self.emg_stats.normalize(EMG)

        #save
        self.save(EMG)
        print("Dataset (un)loading completed successfully")

    @property
    def PEOPLE(self):
        return MAX_PEOPLE_TRAIN if (self.train or self.val) else MAX_PEOPLE_TEST #MAX_PEOPLE 

    @property
    def TASKS(self):
        return MAX_TASKS_TRAIN+1 #if (self.train or self.val) else MAX_TASKS+1

    @property
    def REPS(self):
        if self.train:
            return MAX_TRAIN_REPS-1
        elif self.val:
            return 1
        else:
            return MAX_REPS #MAX_TEST_REPS

    @property
    def tasks_mask(self):
        #tasks_mask=self.tasks_train if (self.train or self.val) else self.tasks
        tasks_mask=self.tasks_train # if (self.train or self.val) else self.tasks
        tasks_mask=torch.cat((tasks_mask, torchize([0])))
        return tasks_mask

    @property
    def people_mask(self):
        # test on all people
        return self.people_train if (self.train or self.val) else self.people_test

    @property
    def rep_mask(self):
        if self.train:
            return self.rep_train
        elif self.val:
            return self.rep_val
        else:
            return self.reps #self.rep_test

    def __len__(self):
        return self.PEOPLE*self.TASKS*self.REPS*WINDOW_OUTPUT_DIM

    def slice_batch(self, DIM, idx):
        task_idx = idx // (self.PEOPLE * self.REPS * WINDOW_OUTPUT_DIM)
        #"""
        people_rep_output_idx = idx % (self.PEOPLE * self.REPS * WINDOW_OUTPUT_DIM)
        people_idx = people_rep_output_idx // (self.REPS * WINDOW_OUTPUT_DIM)
        rep_output_idx = people_rep_output_idx % (self.REPS * WINDOW_OUTPUT_DIM)
        tensor = self.EMG[people_idx, task_idx]
        tensor = tensor.reshape(-1, DIM)[rep_output_idx]
        #"""
        
        # now we have only one slice, convert to an image
        tensor = tensor.reshape(1, 1, DIM)
        label = torchize(task_idx)
        return tensor, label

    def __getitem__(self, idx):
        if self.raw:
            return self.EMG
        EMG, label=self.slice_batch(EMG_DIM, idx)
        return EMG, label


def load(db):
    db.load_dataset()

def info(db):
    print("New tasks", db.tasks_test.cpu().numpy())
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
        batch=len(db)
        print("\tBatch amts: %s"%(batch))
        
def visualize(db):
    import matplotlib.pyplot as plt
    db.raw=True
    db.set_train()
    EMG=db[0]
    dat=EMG[0,0,0,:,:].cpu().numpy()
    dim=EMG_DIM
    for sensor in range(dim):
        print(dat[:, sensor].max(), sensor)
        plt.plot(dat[:, sensor])
    plt.show()
    
if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Loading ninapro dataset')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--info', action='store_true')
    parser.add_argument('--norm', action='store_false')
    args = parser.parse_args()

    db=DB23()
    if args.load:
        load(db)
    if args.viz:
        db.load_stored()
        visualize(db)
    if args.info:
        db.load_stored()
        info(db)
