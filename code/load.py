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
    def __init__(self, adabn=True, train=True, val=False):
        self.device=torch.device("cuda")
        self.train=train
        self.val=val
        self.adabn=adabn
        self.raw=False

        self.tasks_train=torchize(TRAIN_TASKS)
        self.tasks_test=torchize(TEST_TASKS)
        self.tasks=torchize(TASKS)

        self.people_train=torchize(TRAIN_PEOPLE_IDXS)
        self.people=torchize(PEOPLE_IDXS)

        _rand_perm_train=torch.randperm(MAX_TRAIN_REPS, device=self.device)
        _rand_perm_test=torch.randperm(MAX_TEST_REPS, device=self.device)

        train_reps = torchize(TRAIN_REPS)
        test_reps = torchize(TEST_REPS)
        self.rep_train=train_reps[:-1]-1
        self.rep_val=train_reps[-1:]-1
        self.rep_test=test_reps-1

        self.window_mask=torch.randperm(WINDOW_OUTPUT_DIM-1, device=self.device)

        self.path="/home/breezy/ULM/prosthetics/db23/"

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
            reps = int((MAX_TRAIN_REPS-1)*(1/BLOCK_SIZE if self.adabn else 1))
        elif self.val:
            reps = 1
        else:
            reps = int((MAX_TEST_REPS-1)*(1/BLOCK_SIZE if self.adabn else 1))
        windows = MAX_WINDOW_BLOCKS if self.adabn else WINDOW_OUTPUT_DIM
        tasks=1 if self.adabn else self.TASKS
        return self.PEOPLE*reps*windows*tasks

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

    def get_stim_rep(self, stimulus, repetition):
        # stimulus from 1-40, repetition from 1-6
        ex=np.searchsorted(np.array(TASK_DIST).cumsum(), stimulus)
        emg, acc, glove, stim, rep = self.Es[ex]
        # emg, acc, glove, stimulus, repetition
        stim_mask, rep_mask=(stim==stimulus), (rep==repetition)
        mask=(stim_mask&rep_mask).squeeze()

        # this was removed in https://www.nature.com/articles/s41597-019-0349-2 for noise problems as well
        glove=np.concatenate((glove[:, :10], glove[:, 11:]), axis=1)
        emg_=emg[mask][:TOTAL_WINDOW_SIZE+2*WINDOW_EDGE]
        # align data with emg
        acc_=acc[mask][WINDOW_EDGE:TOTAL_WINDOW_SIZE+WINDOW_EDGE]
        glove_=glove[mask][WINDOW_EDGE:TOTAL_WINDOW_SIZE+WINDOW_EDGE]

        # filter raw signal - 20 Hz to 400 Hz
        emg_=filter(emg_*2**10, (20, 400), butterworth_order=4,btype="bandpass") # bandpass filter
        # rectify signal - window of ~147
        emg_=rms(emg_)

        emg_=torchize(emg_[self.time_mask])
        glove_=torchize(glove_[self.time_mask])
        acc_=torchize(acc_[self.time_mask])
        return emg_, acc_, glove_

    def load_dataset(self):
        """
        Loads dataset as a pt file format all preprocessed.
        subject -> reps -> stim -> amt windows -> window_ms -> dim (emg,acc,glove)
        """
        global args
        print("People:", PEOPLE)
        self.time_mask=np.arange(0,TOTAL_WINDOW_SIZE,FACTOR,dtype=np.uint8)
        
        self.emg_stats=RunningStats(norm=args.norm, complete=True)
        self.acc_stats=RunningStats(norm=args.norm)
        self.glove_stats=RunningStats(norm=args.norm)

        shape=(len(PEOPLE), MAX_TASKS+1,MAX_REPS,len(self.time_mask))
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
            # clip tensors of outliers - no clipping for now
            #self.Es=self.clip_es(self.Es)

            for rep in range(0,MAX_REPS):
                for stim in range(MAX_TASKS+1):
                    emg,acc,glove=self.get_stim_rep(stim,rep+1)
                    # only add if in the training set
                    if (person in TRAIN_PEOPLE and rep in self.rep_train and stim in TRAIN_TASKS):
                        self.emg_stats.push(emg) 
                        self.acc_stats.push(acc) 
                        self.glove_stats.push(glove) 

                    EMG[i,stim,rep]=emg
                    ACC[i,stim,rep]=acc
                    GLOVE[i,stim,rep]=glove 

        if args.norm:
            emg_means,emg_std=self.emg_stats.mean_std()
            acc_means,acc_std=self.acc_stats.mean_std()
            glove_means,glove_std=self.glove_stats.mean_std()
            print(emg_std,acc_std,glove_std)
        else:
            print("Min max normalization:\n")
            print("\tEMG min max:", self.emg_stats.min.item(), self.emg_stats.max.item())
            print("\tACC min max:", self.acc_stats.min.item(), self.acc_stats.max.item())
            print("\tGLOVE min max:", self.glove_stats.min.item(), self.glove_stats.max.item())

        # normalize
        EMG=self.emg_stats.normalize(EMG)
        ACC=self.acc_stats.normalize(ACC)
        GLOVE=self.glove_stats.normalize(GLOVE)

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
        tasks_mask=self.tasks_train if (self.train or self.val) else self.tasks
        tasks_mask=torch.cat((tasks_mask, torchize([0])))
        return tasks_mask

    @property
    def people_mask(self):
        return self.people_train if (self.train or self.val) else self.people

    @property
    def block_mask(self):
        if self.train:
            return self.rep_train
        elif self.val:
            return self.rep_val
        else:
            return self.rep_test

    def get_idx_(self, idx):
        # larger batch size in training
        mul = self.PEOPLE * MAX_WINDOW_BLOCKS
        rep_block = idx // mul
        sub_wind_idx = idx % mul
        subject = sub_wind_idx // MAX_WINDOW_BLOCKS
        window_block = sub_wind_idx % MAX_WINDOW_BLOCKS
        return (rep_block, subject, window_block)

    def slice_batch(self, tensor, tmask, bmask, wmask, dim):
        shape=(self.TASKS*BLOCK_SIZE, FINAL_WINDOW_SIZE, dim)
        tensor=tensor[tmask, bmask].reshape(shape)
        stride = (FINAL_WINDOW_SIZE*dim, WINDOW_STRIDE*dim, dim, 1)
        tensor=torch.as_strided(tensor, (self.TASKS*BLOCK_SIZE, WINDOW_OUTPUT_DIM-1, WINDOW_MS, dim), stride)
        return tensor[:, wmask].reshape(-1, 1, WINDOW_MS, dim)

    def slice_idx(self, tensor, dim, idx):
        t=tensor.transpose(0,1)[self.tasks_mask]
        t=t[:, self.people_mask]
        t=t[:, :, self.block_mask]
        return t.reshape(-1, 1, WINDOW_MS, dim)[idx]

    def __getitem__(self, batch_idx):
        # batch_idx -> rep_block x (subject x window_block)
        # return batch of shape ( rep * amtwindows * maxtask) x windowms x respective dim
        # acc dimension is just a mean
        # glove and emg are images (windowms x respective dim)
        # shape is (people, tasks, reps, finalwindowsize, dim)
        if not self.adabn:
            # randomly sample everything - forget about strides (fix later)
            # always 37 tasks though
            # EMG transpose becomes (tasks, people, reps, ...)
            # output shape = (37, 1, ms, dim)
            emg=self.slice_idx(self.EMG,EMG_DIM,batch_idx)
            glove=self.slice_idx(self.GLOVE,GLOVE_DIM,batch_idx)
            acc=self.slice_idx(self.ACC,ACC_DIM,batch_idx).mean(2)
            return (emg,glove,acc)

        if self.raw:
            return self.EMG,self.ACC,self.GLOVE
        rep_block, subject, window_block = self.get_idx_(batch_idx)

        block_mask=self.block_mask[rep_block*BLOCK_SIZE:(rep_block+1)*BLOCK_SIZE]
        window_mask=self.window_mask[window_block*WINDOW_BLOCK:(window_block+1)*WINDOW_BLOCK]
        tasks_mask=self.tasks_mask

        # from people mask
        subject = self.people_mask[subject]
        # from the actual big tensor
        subject = (self.people == subject).nonzero(as_tuple=False).item()

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
    EMG,GLOVE,ACC=db[0]
    dat=EMG.squeeze(0).expand(100,-1).cpu().numpy()
    dim=EMG_DIM
    for sensor in range(dim):
        plt.plot(dat[:, sensor])
    plt.show()
    
if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Loading ninapro dataset')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--info', action='store_true')
    parser.add_argument('--norm', action='store_false')
    args = parser.parse_args()

    db=DB23(adabn=False)
    if args.load:
        load(db)
    if args.viz:
        db.load_stored()
        visualize(db)
    if args.info:
        db.load_stored()
        info(db)
