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
import matplotlib.pyplot as plt
import pyxis as px
try:
     set_start_method('spawn')
except RuntimeError:
    pass

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

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

        # own little dataset
        self.glover=Glover()

    def set_train(self):
        self.train=True
        self.val=False
        self.load_valid()

    def set_val(self):
        self.train=False
        self.val=True
        self.load_valid()

    def set_test(self):
        self.train=False
        self.val=False
        self.load_valid()


    def load_stored(self):
        self.EMG=torch.load(PATH_DIR+'data/emg.pt', map_location=self.device)
        self.EMG.cuda()
        # transpose for indexing
        # people x tasks x ... -> tasks x people x ...
        self.EMG=self.EMG.transpose(0,1)
        print("Loading stored emg...", self.EMG.shape)
        self.GLOVE=self.glover.load_stored()

    def save(self, tensor):
        torch.save(tensor, PATH_DIR+'data/emg.pt')

    def get_np(self, dbnum, p_dir, n):
       E_mat=sio.loadmat("../db%s/s%s/S%s_E%s_A1"%(dbnum,p_dir,p_dir,n))
       emg=E_mat['emg']
       stimulus=E_mat['restimulus']
       repetition=E_mat['rerepetition']
       return emg, stimulus, repetition


    def get_stim_rep(self, stimulus, repetition):
        # stimulus from 1-40, repetition from 1-6
        ex=np.searchsorted(TASK_DIST.cumsum(), stimulus)
        emg, stim, rep = self.Es[ex]
        # emg, stimulus, repetition
        stim_mask, rep_mask=(stim==stimulus), (rep==repetition)
        mask=(stim_mask&rep_mask).squeeze()

        emg_=emg[mask][:TOTAL_WINDOW_SIZE+2*WINDOW_EDGE]

        # filter raw signal - 20 Hz to 450 Hz
        emg_=filter(emg_*2**10, (20, 450), butterworth_order=4,btype="bandpass") # bandpass filter
        # rectification might be problematic for real time software
        emg_=rms(emg_)
        #emg_=np.abs(emg_[WINDOW_EDGE:TOTAL_WINDOW_SIZE+WINDOW_EDGE])
        emg_=torchize(emg_[self.time_mask])
        return emg_

    def load_dataset(self, glove=False):
        """
        Loads dataset as a pt file format all preprocessed.
        subject -> reps -> stim -> amt windows -> window_ms -> dim (emg)
        """
        if glove:
            self.glover.load_dataset()
            return

        global args
        print("People:", PEOPLE)
        print("Tasks (not including rest):" ,TASKS)
        self.time_mask=np.arange(0,TOTAL_WINDOW_SIZE,FACTOR,dtype=np.uint8)
        self.emg_stats=RunningStats(PATH_DIR+"data/emg_", complete=args.complete)
        shape=(len(PEOPLE),MAX_TASKS,MAX_REPS,len(self.time_mask))
        EMG=torch.empty(shape+(EMG_DIM,), device=self.device)

        for i, person in enumerate(tqdm(PEOPLE)):
            # gestures go from 1 to 17, 1 to 23, rest (0)
            # emg stimulus, repetition
            self.person=person
            dbnum="3" if person >= MAX_PEOPLE_D2 else "2"
            subject=person
            if dbnum=="3":
                subject %= MAX_PEOPLE_D2
            p_dir=str(subject+1)

            # Fetch from .mat file
            E1=self.get_np(dbnum,p_dir,"1")
            E2=self.get_np(dbnum,p_dir,"2")
            self.Es = (E1, E2)

            for rep in range(MAX_REPS):
                for stim in range(MAX_TASKS):
                    emg=self.get_stim_rep(stim,rep+1)
                    # only add if in the training set
                    if (person in TRAIN_PEOPLE and rep in self.rep_train and (stim in TRAIN_TASKS or stim==0)):
                        self.emg_stats.push(emg) 
                    EMG[i,stim,rep]=emg

        emg_means,emg_std=self.emg_stats.mean_std()
        print(emg_std)

        # normalize
        EMG=self.emg_stats.normalize(EMG)
        print("Emg shape:", EMG.shape)
        #save
        self.save(EMG)
        print("Dataset (un)loading completed successfully")
        
        if not args.no_glove:
            # now glover's turn
            self.glover.load_dataset()
            
    @property
    def tasks_mask(self):
        #tasks_mask=self.tasks_train if (self.train or self.val) else self.tasks
        #tasks_mask=self.tasks_train if (self.train or self.val) else self.tasks_test[:1]
        # TODO: change back to all tasks
        tasks_mask=self.tasks_train
        tasks_mask=torch.cat((tasks_mask, torchize([0])))
        return tasks_mask

    @property
    def people_mask(self):
        return self.people_train if (self.train or self.val) else self.people_test
        # same subject classification
        #return torchize([41])

    @property
    def rep_mask(self):
        if self.train:
            return self.rep_train
        elif self.val:
            return self.rep_val
            #return self.rep_train
        else:
            return self.rep_test

    @property
    def PEOPLE(self):
        return len(self.people_mask)

    @property
    def TASKS(self):
        return len(self.tasks_mask)

    @property
    def REPS(self):
        return len(self.rep_mask)
        
    @property
    def D(self):
        return self.PEOPLE*self.REPS*self.OUTPUT_DIM


    @property
    def OUTPUT_DIM(self):
        if self.train:
            return int(WINDOW_OUTPUT_DIM)
        return int(PREDICTION_WINDOW_SIZE)

    def load_valid(self):
        tensor=self.EMG[self.tasks_mask][:, self.people_mask][:, :, self.rep_mask]
        tensor=tensor[:, :, :, :self.OUTPUT_DIM]
        # tasks x people x rep -> tasks*(people*rep*output_dim)
        self.EMG_use=tensor.reshape(-1, EMG_DIM)
        a=self.EMG_use[self.D*2+1]
        b=tensor[2].reshape(-1,EMG_DIM)[1]
        assert torch.equal(a, b), "indexing is not correct"
        self.glover.load_valid(self.tasks_mask)

    def __len__(self):
        return self.TASKS*self.D

    def slice_batch(self, idx):
        # every self.D is a new task
        tensor = self.EMG_use[idx].reshape(-1, 1, 1, EMG_DIM)
        return tensor

    def __getitem__(self, idx):
        if self.raw:
            return self.EMG
        EMG=self.slice_batch(idx)
        return EMG

def load(db,glove=False):
    db.load_dataset(glove=glove)

def info(db):
    print("New tasks", db.tasks_test.cpu().numpy())
    for train in [False, True]:
        if train:
            db.set_train()
            e,label=db[0]
            print(e.min(), e.max())
        else:
            db.set_test()
            e,label=db[0]
            print(e.min(), e.max())
        print("TRAIN:" if train else "TEST:")
        batch=len(db)
        print("\tBatch amts: %s"%(batch))
        
def visualize(db):
    db.set_train()
    dat=db.EMG[args.person, args.task,args.rep,:,:].cpu().numpy()
    for sensor in range(EMG_DIM):
        plt.plot(dat[:, sensor])
    plt.show()
    
if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Loading ninapro dataset')
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--rep', type=int, default=0)
    parser.add_argument('--person', type=int, default=0)
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--load_glove', action='store_true')
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--info', action='store_true')
    parser.add_argument('--complete', action='store_true')
    parser.add_argument('--no_glove', action='store_true')
    args = parser.parse_args()

    db=DB23()
    if args.load:
        load(db)
    db.load_stored()
    if args.load_glove:
        load(db, glove=True)
    if args.viz:
        visualize(db)
    if args.info:
        info(db)
