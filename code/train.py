import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.utils.data as data
from dataloader import DB23
from constants import *
import tqdm 
from models import GLOVENet, EMGNet

torch.manual_seed(42)

# modeled after https://github.com/openai/CLIP/blob/main/clip/model.py
class Model(nn.Module):
    def __init__(self, d_e, train_model=True, device="cuda"):
        super(Model,self).__init__()

        self.train_model = train_model
        self.d_e = d_e
        self.device = torch.device(device)

        self.emg_net = EMGNet(d_e=d_e) # emg model
        self.glove_net = GLOVENet(d_e=d_e) # glove model

        self.loss_f = torch.nn.functional.cross_entropy

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1))#/0.07))    # CLIP logit scale

        self.to(self.device)

    def set_train(self):
        self.train_model=True

    def set_test(self):
        self.train_model=False

    def set_val(self):
        self.set_test() # same behavior

    def encode_emg(self, EMG, ACC):
        emg_features = self.emg_net(EMG, ACC)
        return emg_features

    def encode_glove(self, GLOVE):
        glove_features = self.glove_net(GLOVE)
        return glove_features

    def forward(self, EMG, ACC, GLOVE, EMG_T, GLOVE_T):

        # only for randnet
        self.emg_net.T=EMG_T
        self.glove_net.T=GLOVE_T

        #emg_features = self.encode_emg(EMG, ACC).reshape((-1,EMG_T,self.d_e))
        #glove_features = self.encode_glove(GLOVE).reshape((-1,GLOVE_T,self.d_e))
        emg_features = self.encode_emg(EMG, ACC)
        #print(emg_features.shape, EMG.shape)
        emg_features = emg_features.reshape((EMG_T,-1,self.d_e)).permute((1,0,2))
        #ef = self.encode_emg(EMG, ACC).reshape((-1,EMG_T,self.d_e))
        #print(ef.flatten()[-3:],emg_features.flatten()[-3:])

        glove_features = self.encode_glove(GLOVE)
        glove_features = glove_features.reshape((GLOVE_T,-1,self.d_e)).permute((1,0,2))
        

        emg_features = emg_features / emg_features.norm(dim=-1,keepdim=True)
        glove_features = glove_features / glove_features.norm(dim=-1,keepdim=True)

        #                        -> N_e x N_g
        # encoders give input as (TASKS*WINDOW_BLOCK, d_e) or another N
        # we want (-1, TASKS, d_e) and take cross entropy across entire (-1) dim

        logit_scale=self.logit_scale.exp().clamp(min=1e-8,max=100)
        logits = torch.matmul(emg_features, glove_features.permute(0,2,1)) # shape = (N,TASKS_e,TASKS_g)

        if self.train_model:
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
        return self.glove_probs(logits).argmax(dim=2) # (N,tasks_e), glove pred from each emg

    def predict_emg_from_glove(self, logits):
        return self.emg_probs(logits).argmax(dim=2) # (N,tasks_g), emg pred for each glove

    def correct_glove(self, logits, vote=True):
        N,tasks,tasks=logits.shape

        argmax_glove=self.predict_glove_from_emg(logits)
        labels_=self.get_labels(N,tasks)
        correct = (argmax_glove.reshape(-1)==labels_).sum().cpu().item()

        if not vote:
            return correct
        else:
            # (tasks), mean logits over the entire thing
            argmax_glove=self.predict_glove_from_emg(logits.mean(0).unsqueeze(0)).flatten()
            labels=self.get_labels(1,tasks).flatten()
            correct_maj = (argmax_glove==labels).sum().cpu().item()

        return correct_maj, correct

    # TODO: copy from above
    """
    def correct_emg(self, logits,vote=True):
        N,tasks,tasks=logits.shape
        argmax_emg=self.predict_emg_from_glove(logits)

        if not vote:
            labels=self.get_labels(N,tasks)
            return (argmax_emg.reshape(-1)==labels).sum().cpu().item()

        argmax_emg=argmax_emg.mode(0)    # voting across 150 ms
        labels=self.get_labels(1,tasks).reshape(-1)
        return (argmax_emg==labels).sum().cpu().item()
    """

    def get_labels(self, N, tasks):
        return torch.stack([torch.arange(tasks,dtype=torch.long,device=self.device)]*N).reshape(N*tasks)

    def loss(self, logits):

        # matrix should be symmetric
        N,tasks,tasks=logits.shape  # e x g
        labels = self.get_labels(N,tasks)
        # convert (N_e, N_g) -> (n,task_e,N_g) -> (n,task_e,n,task_g) -> (n,n,task_g,task_e) -> (n^2,task_g,task_e)
        logits_e = logits.reshape((N*tasks,tasks))
        logits_g = logits.permute((0,2,1)).reshape((N*tasks,tasks))
        loss_e = self.loss_f(logits_e, labels,reduction='mean')
        loss_g = self.loss_f(logits_g, labels,reduction='mean')
        loss = (loss_e+loss_g)/2
        return loss

    def l2(self):
        return self.emg_net.l2() + self.glove_net.l2()

def test(model, dataset):
    dataset.set_test()
    model.set_test()
    model.eval()

    test_loader=data.DataLoader(dataset=dataset,batch_size=1,shuffle=True,num_workers=NUM_WORKERS)
    total_tasks=dataset.TASKS
    total_loss = []
    total_correct = []  # in each 15 ms
    total_correct_voting = [] # in all
    total=0

    logits_labels = []   # rep_block, subject, window block

    for i, (EMG,GLOVE,ACC) in enumerate(test_loader):
        cnt=total//EMG.shape[0]
        EMG,ACC,GLOVE=EMG.squeeze(0),ACC.squeeze(0),GLOVE.squeeze(0)
        logits = model.forward(EMG,ACC,GLOVE,total_tasks,total_tasks)

        # take mean for all 15 ms windows
        logits_save = logits.mean(0).detach().cpu().numpy()
        np.save('../data/logits_%d.npy' % i, logits_save)

        logits_labels.append(dataset.get_idx_(i))

        loss=model.loss(logits)
        total_loss.append(loss.item())

        correct_maj,correct=model.correct_glove(logits)

        total_correct.append(correct)
        total_correct_voting.append(correct_maj)
        total+=EMG.shape[0]

    total_loss=np.array(total_loss)
    logits_labels=np.array(logits_labels)
    np.save('../data/logits_labels.npy', logits_labels)

    print("Testing finished, logits saved, max %d"%(len(test_loader)-1))

    return total_loss.mean(), sum(total_correct)/total, sum(total_correct_voting)/total


def validate(model, dataset):
    dataset.set_val()
    model.set_val()
    model.eval()

    val_loader=data.DataLoader(dataset=dataset,batch_size=1,shuffle=True,num_workers=NUM_WORKERS)
    total_tasks=dataset.TASKS
    total_loss = []
    total_correct = []
    total_correct_maj = []
    total=0

    for (EMG,GLOVE,ACC) in val_loader:
        EMG,ACC,GLOVE=EMG.squeeze(0),ACC.squeeze(0),GLOVE.squeeze(0)
        logits = model.forward(EMG,ACC,GLOVE,total_tasks,total_tasks)
        loss=model.loss(logits)
        total_loss.append(loss.item())
        correct_maj, correct=model.correct_glove(logits)
        total_correct_maj.append(correct_maj)
        total_correct.append(correct)
        total+=EMG.shape[0]

    total_loss=np.array(total_loss)
    model.set_train()
    return total_loss.mean(), sum(total_correct)/total, sum(total_correct_maj)/total

def train_loop(dataset, train_loader, params, checkpoint=False,checkpoint_dir="../checkpoints/model", annealing=False,load=None):

    # crossvalidation parameters
    model = Model(d_e=params['d_e'], train_model=True)
    if load is not None:
        print("Loading model")
        model.load_state_dict(torch.load(checkpoint_dir+"%d.pth"%load))
    # little higher for annealing
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=0)
    if annealing:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0,verbose=True)
        
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params['step_size'], gamma=params['gamma'])
    #TODO: Load and save model parameters

    # train data
    dataset.set_train()
    total_tasks = dataset.TASKS

    val_losses={}
    counter=0

    print("Training...")
    for e in tqdm.trange(params['epochs']):
        loss_train=[]
        for (EMG,GLOVE,ACC) in train_loader:
            EMG,ACC,GLOVE=EMG.squeeze(0),ACC.squeeze(0),GLOVE.squeeze(0)
            loss=model.forward(EMG,ACC,GLOVE,total_tasks,total_tasks)+model.l2()*params['l2']
            optimizer.zero_grad()
            loss.backward()
            loss_train.append((loss-model.l2()*params['l2']).item())
            optimizer.step()

        scheduler.step()
        loss_val,acc,acc_maj=validate(model, dataset)
        loss_train=np.array(loss_train).mean()
        print("Current epoch %d, train_loss: %.4f, val loss: %.4f, acc: %.6f, acc_maj: %.4f" % (e, loss_train, loss_val, acc, acc_maj))

        final_val_acc=(loss_val,acc,acc_maj)
        val_losses[e]=acc_maj   # TODO: acc_maj >= acc?

        if checkpoint and acc >= max(list(val_losses.values())):
            torch.save(model.state_dict(), checkpoint_dir+"%d.pt"%counter)
            counter+=1

    return final_val_acc, model

def main():
    # Do not make new object for train=False just change (randomization would change)
    new_people=3
    new_tasks=4
    dataset23 = DB23(new_tasks=new_tasks,new_people=new_people) # new_people are amputated
    train_loader=data.DataLoader(dataset=dataset23,batch_size=1,shuffle=True,num_workers=NUM_WORKERS)
    # parameters

    
    #
    #$lrs=[0.003981071705534973]
    #regs=[0.00031622776601683794]

    #"""
    lrs = np.logspace(-6,0,num=20)
    regs = np.logspace(-8,2,num=12) # -8,-3
    des=[128,256]

    print(lrs)
    print(regs)
    print(des)
    cross_val={}

    for d_e in des:
        for lr in lrs:
            for reg in regs:
                print("d_e: %s, lr: %s, reg: %s"%(str(d_e),str(lr),reg))
                params = {
                        'd_e' : d_e,
                        'epochs' : 6,
                        'lr' : lr,
                        'step_size' : 10,
                        'gamma' : 1,
                        'l2' : reg
                        }

                (loss_t,acc_t,acc_maj),model=train_loop(dataset23, train_loader, params, checkpoint=True)
                cross_val[(d_e, lr, reg)]=(acc_maj, acc_t, loss_t) #TODO

        print(cross_val)
        print(sorted(list(cross_val.values()),reverse=True))
    print(cross_val)
    vals = np.array(list(cross_val.values()))
    keys = np.array(list(cross_val.keys()))
    print(vals.sort())
    #"""

    """

    #d_e, lr, reg = keys[vals.argmax()]         # fix this, incorrect
    #checkpoint=878
    """

    """
    checkpoint=10
    d_e, lr, reg = 128, 0.0012689610031679222, 1e-6
    # lr: 0.001, reg: 1e-06
    #lr = 5e-4
    lr = 1e-3
    #reg=1e-6
    reg = 1e-4

    print("Final train")
    params = {
            'd_e' : d_e,
            'epochs' : 20_000,
            'lr' : lr,
            'step_size' : 10,
            'gamma' : 1,
            'l2' : reg
            }
    final_vals, model=train_loop(dataset23, train_loader, params, checkpoint=True,annealing=True,load=checkpoint)
    final_stats=test(model, dataset23)
    print(final_vals)
    print(final_stats)
    
   """


if __name__=="__main__":
    main()
