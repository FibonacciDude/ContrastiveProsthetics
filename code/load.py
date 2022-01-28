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

        self.reps=torchize(REPS)-1
        self.reps_train=torchize(REPS_TRAIN)-1
        self.reps_val=torchize(REPS_VAL)-1

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
        #self.EMG=self.EMG.transpose(0,1)
        print("Loading stored emg...", self.EMG.shape)
        self.GLOVE=self.glover.load_stored()

    def save(self, tensor):
        torch.save(tensor, PATH_DIR+'data/emg.pt')

    def get_np(self, dbnum, p_dir, n):
       E_mat=sio.loadmat("../db%s/s%s/S%s_E%s_A1"%(dbnum,p_dir,p_dir,n))
       stimulus=E_mat['restimulus'][::FACTOR]
       repetition=E_mat['rerepetition'][::FACTOR]
       emg=E_mat['emg']

       # do all preprocessing first (this adds a delay to the labels by a little, it should mostly be okay)
       emg=filter(emg, 25, butterworth_order=1, btype="highpass")
       emg=filter(emg, 1, butterworth_order=1, btype="lowpass")
       # downsample second - doing it after stim/rep batch results in uneven samples
       emg=emg[::FACTOR]
       return emg, stimulus, repetition


    def get_stim_rep(self, stimulus, repetition):
        # stimulus from 1-40, repetition from 1-6
        ex=np.searchsorted(TASK_DIST.cumsum(), stimulus)
        emg, stim, rep = self.Es[ex]
        # emg, stimulus, repetition
        stim_mask, rep_mask=(stim==stimulus), (rep==repetition)
        mask=(stim_mask&rep_mask).squeeze()

        # move over by some in order to not have another stim/rep
        # tweak window size compared to db1
        emg_=emg[mask][:FINAL_WINDOW_SIZE+2*WINDOW_EDGE]
        emg_=rms(emg_)

        # normalize - this is not the same as the one done for ninapro db1 (change)
        #if not args.no_minmax:
        #   a=2.5
        #   emg_=(emg_+a*10**-6)/(2*(a*10**-6))-.5
        #else:
        #u=2048
        #emg_=np.sign(emg_) * np.log(1+u*np.abs(emg_))/np.log(1+u)
        #emg_=emg_[WINDOW_EDGE:-WINDOW_EDGE]

        if args.debug:
            for sensor in range(EMG_DIM):
                plt.plot(emg_[:, sensor])
            plt.show()

        emg_=torchize(emg_)
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
        self.time_mask=np.arange(FINAL_WINDOW_SIZE,dtype=np.uint8)
        if args.no_minmax:
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
                    if args.no_minmax:
                        if ((person in TRAIN_PEOPLE) and (rep in self.reps_train) and (stim in TRAIN_TASKS or stim==0)):
                            self.emg_stats.push(emg)
                    EMG[i,stim,rep]=emg

        if args.no_minmax:
            # normalize
            emg_means,emg_std=self.emg_stats.mean_std()
            print(emg_std)
            EMG=self.emg_stats.normalize(EMG)

        else:
            emg=EMG[self.people_mask]
            emg=emg[:, self.tasks_mask]
            emg=emg[:, :, self.rep_mask]

            vals,idxs=emg.flatten().sort()
            max_val=vals[int(.95*emg.flatten().shape[0])]
            min_val=vals[int(.05*emg.flatten().shape[0])]
            min_val,max_val=emg.min(),emg.max()
            EMG=(EMG-min_val)/(max_val-min_val)

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
        if self.train:
            return self.people_train
        elif self.val:
            return self.people_train
        else:
            return self.people_test

    @property
    def rep_mask(self):
        if self.train:
            return self.reps_train
        elif self.val:
            return self.reps_val
        else:
            return self.reps

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
        #return self.PEOPLE*self.REPS*self.OUTPUT_DIM
        # plus people
        #return self.REPS*(self.OUTPUT_DIM-WINDOW_MS+1)
        return self.REPS*(self.OUTPUT_DIM//WINDOW_MS)

    @property
    def OUTPUT_DIM(self):
        if self.train:
            return int(WINDOW_OUTPUT_DIM)
        return int(PREDICTION_WINDOW_SIZE)

    def load_valid(self):
        tensor=self.EMG[self.people_mask][:, self.tasks_mask]
        tensor=tensor[:, :, self.rep_mask][:, :, :, :self.OUTPUT_DIM][:, :, :, :, :EMG_DIM]
         
        # windows
        tensor=tensor.reshape(self.PEOPLE, self.TASKS, self.REPS, self.OUTPUT_DIM//WINDOW_MS, WINDOW_MS, EMG_DIM)

        # TODO: write tests (when it seems like it works)
    
        #tensor=tensor.reshape(self.PEOPLE, -1, self.OUTPUT_DIM, EMG_DIM)
        #shape=tensor.shape
        #stride=(np.prod(shape[1:]), np.prod(shape[2:]), WINDOW_STRIDE*EMG_DIM, EMG_DIM, 1)
        #tensor=torch.as_strided(tensor, size=(self.PEOPLE, shape[1], self.OUTPUT_DIM-WINDOW_MS+1, WINDOW_MS, EMG_DIM), stride=stride)

        # people x tasks x reps x outputdim x dim
        self.tensor=tensor.reshape(self.PEOPLE, -1, WINDOW_MS, EMG_DIM)
        self.glover.load_valid(self.tasks_mask)

    def __len__(self):
        return self.TASKS*self.D

    def slice_batch(self, people_idx, idx):
        # idx can be an array, but not people_idx
        # every self.D is a new task
        tensor = self.tensor[people_idx][idx].reshape(-1, 1, WINDOW_MS, EMG_DIM)
        self.domain=self.people_mask[people_idx]      # the actual person
        return tensor

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
    #dat=db.EMG[args.person, args.task,args.rep,:,:].cpu().numpy()
    dat=db.slice_batch(args.person, 0).reshape(WINDOW_MS, EMG_DIM).cpu().numpy()
    print(dat)
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
    parser.add_argument('--no_minmax', action='store_true')
    parser.add_argument('--debug', action='store_true')

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
