import torch
import torch.nn.utils as utils
import scipy.io.loadmat as loadmat
from constants import *

torch.manual_seed(42)

class DB23(utils.Dataset):
    def __init__(self, new_people=2, new_tasks=5, train=True, device="cuda"):
        self.device=torch.device(device)
        self.train=train

        self.NEW_PEOPLE=new_people
        self.NEW_TASKS=new_tasks
        self.MAX_PEOPLE_TRAIN=MAX_PEOPLE-new_people

        self.people_rand_d2=torch.randperm(PEOPLE_D2, device=self.device)
        self.people_rand_d3=torch.randperm(PEOPLE_D3, device=self.device)
        self.block_rand_train=torch.randperm(TRAIN_REPS, device=self.device)
        self.block_rand_test=torch.randperm(TEST_REPS, device=self.device)


    def __len__(self):
        if train:
            length=self.MAX_PEOPLE_TRAIN*(MAX_TASKS-self.NEW_TASKS)*TRAIN_REPS
        else:
            length=self.MAX_PEOPLE_TRAIN*(MAX_TASKS-self.NEW_TASKS)*TEST_REPS+(MAX_PEOPLE)*MAX_TASKS*MAX_REPS
        return length

    def __getitem__(self, batch_idx):
        # batch of constant size of tasks*2 (train time)
        # batch_idx is just blck x people
        person = batch_idx%(self.MAX_PEOPLE_TRAIN)
        dbnum="3" if person >= PEOPLE_D2 else "2"
        if dbnum=="2":
            person %= MAX_PEOPLE_D2
        p_dir=str(person+1)
        E1_mat=loadmat("../db%s/s%s/S%s_E%s_A1"%(dbnum,p_dir,p_dir,"1")
        E2_mat=loadmat("../db%s/s%s/S%s_E%s_A1"%(dbnum,p_dir,p_dir,"2")
        E3_mat=loadmat("../db%s/s%s/S%s_E%s_A1"%(dbnum,p_dir,p_dir,"3")
        print(E1_mat['emg'])
        dt=E1_mat['restimulus']
        print(dt.min(),dt.max())

db=DB23()
print(len(db))
print(db[0])
