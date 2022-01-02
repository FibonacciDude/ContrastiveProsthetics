import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.utils.data as data
from dataloader import DB23
from constants import *
import tqdm 

class randnet(nn.Module):
    def __init__(self, d_e, train=True, device="cuda"):
        super(randnet,self).__init__()
        self.device=device
        self.d_e=d_e

    def forward(self, X):
        # encoders give input as (BLOCK_SIZE*TASKS*WINDOW_OUTPUT_DIM, d_e)
        out=torch.rand((BLOCK_SIZE*self.T*WINDOW_OUTPUT_DIM,self.d_e), device=self.device)
        return out

# modeled after https://github.com/openai/CLIP/blob/main/clip/model.py
class Model(nn.Module):
    def __init__(self, d_e, train=True, device="cuda"):
        super(Model,self).__init__()

        self.train = train
        self.d_e = d_e
        self.device = torch.device(device)

        self.emg_net = randnet(d_e=d_e) # emg model
        self.glove_net = randnet(d_e=d_e) # glove model

        self.loss_f = torch.nn.functional.cross_entropy

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))    # CLIP logit scale

        self.to(self.device)

    def set_train(self):
        self.train=True

    def set_test(self):
        self.train=False

    def encode_emg(self, EMG):
        emg_features = self.emg_net(EMG)
        return emg_features

    def encode_glove(self, GLOVE, ACC):
        glove_features = self.glove_net((GLOVE,ACC))
        return glove_features

    def forward(self, EMG, ACC, GLOVE, EMG_T, GLOVE_T):

        # only for randnet
        self.emg_net.T=EMG_T
        self.glove_net.T=GLOVE_T

        emg_features = self.encode_emg(EMG).reshape((-1,EMG_T,self.d_e))
        glove_features = self.encode_glove(GLOVE,ACC).reshape((-1,GLOVE_T,self.d_e))
        emg_features = emg_features / emg_features.norm(dim=-1,keepdim=True)
        glove_features = glove_features / glove_features.norm(dim=-1,keepdim=True)

        #                        -> N_e x N_g
        # encoders give input as (BLOCK_SIZE*TASKS*WINDOW_OUTPUT_DIM, d_e) or another N
        # we want (-1, TASK+1, d_e) and take cross entropy across entire (-1) dim

        logit_scale=self.logit_scale.exp().clamp(min=1e-8,max=100)
        logits = torch.matmul(emg_features, glove_features.permute(0,2,1)) # shape = (N,TASKS_e,TASKS_g)

        if self.train:
            return self.loss(logits * logit_scale)

        return logits

    def glove_probs(self, logits):
        # (N,tasks_e,tasks_g)
        return (100.0 * logits).softmax(dim=-1)

    def emg_probs(self, logits):
        logits = logits.permute((0,2,1))
        # (N,tasks_g,tasks_e)
        return (100.0 * logits).softmax(dim=-1)

    def predict_glove_from_emg(self, logits):
        # glove_probs gives (N,tasks_e,tasks_g)
        return self.glove_probs(logits).argmax(-1) # (N,tasks_e), glove pred from each emg

    def predict_emg_from_glove(self, logits):
        return self.emg_probs(logits).argmax(-1) # (N,tasks_g), emg pred for each glove

    def correct_glove(self, logits):
        N,tasks=logits.shape
        argmax_glove=self.predict_glove_from_emg(logits)
        labels=self.get_labels(N,tasks)
        return (argmax_glove==labels).sum()

    def correct_emg(self, logits):
        N,tasks=logits.shape
        argmax_emg=self.predict_emg_from_glove(logits)
        labels=self.get_labels(N,tasks)
        return (argmax_emg==labels).sum()

    def get_labels(N, tasks):
        return torch.stack([torch.arange(tasks)]*N, dtype=torch.long, device=self.device).reshape(N*tasks, tasks)

    def loss(self, logits):

        # matrix should be symmetric
        N,tasks,tasks=logits.shape  # e x g
        labels = self.get_labels(N,tasks)
        labels = labels.reshape(N*tasks, tasks)
        # convert (N_e, N_g) -> (n,task_e,N_g) -> (n,task_e,n,task_g) -> (n,n,task_g,task_e) -> (n^2,task_g,task_e)
        logits_e = logits.reshape((N*tasks,tasks))
        logits_g = logits.transpose(0,2,1).reshape((N*tasks,tasks))
        loss_e = self.loss_f(logits_e, labels,reduction='mean')
        loss_g = self.loss_f(logits_g, labels,reduction='mean')
        loss = (loss_e+loss_g)/2
        return loss

def validate(model, dataset):
    dataset.set_val()
    model.set_val()
    val_loader=data.DataLoader(dataset=dataset,batch_size=1,shuffle=True,num_workers=NUM_WORKERS)
    total_tasks=dataset.TASKS
    total_loss = []
    total_correct = []
    total=0

    for i, (EMG,ACC,GLOVE) in enumerate(val_loader):
        logits = model.forward(EMG,ACC,GLOVE,total_tasks,total_tasks)
        loss=model.loss(logits)
        total_loss.append(loss)
        del EMG,ACC,GLOVE
        correct=model.correct_glove(logits)
        total_correct.append(correct)
        total+=EMG.shape[0]

    total_loss=np.array(total_loss)
    model.set_train()
    return total_loss.mean(), sum(total_correct)/(total)

def train_loop(dataset, train_loader, params):

    global  device_location
    # crossvalidation parameters
    model = Model(d_e=params['d_e'], train=True, device=device_location)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params['step_size'], gamma=params['gamma'])
    #TODO: Load and save model parameters

    # train data
    dataset.set_train()
    total_tasks = dataset.TASKS

    for e in tqdm.trange(params['epochs']):
        for i, (EMG,ACC,GLOVE) in enumerate(train_loader):
            loss=model.forward(EMG,ACC,GLOVE,total_tasks,total_tasks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        acc,loss=validate(model, dataset)
        print("Current epoch %d loss: %.4f, acc: %.4f" % (e, loss,acc))

device_location = "cpu"

def main():
    global device_location
    # Do not make new object for train=False just change (randomization would change)
    dataset23 = DB23(new_tasks=4,new_people=3, device=device_location) # new_people are amputated
    train_loader=data.DataLoader(dataset=dataset23,batch_size=1,shuffle=True,num_workers=NUM_WORKERS)
    # parameters
    new_people=3
    new_tasks=4

    params = {
            'device' : "cpu",
            'd_e' : 128,
            'epochs' : 1,
            'lr' : 1e-2,
            'step_size' : 4,
            'gamma' : .2
            }

    train_loop(dataset23, train_loader, params)

if __name__=="__main__":
    main()
