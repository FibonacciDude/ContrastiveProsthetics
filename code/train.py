import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.utils.data as data
from dataloader import DB23
from constants import *
import tqdm 
from models import GLOVENet, EMGNet, Model
from pprint import pprint
import json
import time
import torch.cuda.amp as amp

torch.manual_seed(42)
torch.backends.cudnn.benchmark=True

def test(model, dataset):
    dataset.set_test()
    model.set_test()
    model.eval()

    idx_test = torch.randperm(len(dataset))
    loader=data.DataLoader(dataset, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=PREFETCH)
    print("Test size:", len(dataset))

    total_tasks=dataset.TASKS
    total_loss = []
    total_correct = []  # in each 15 ms
    total=0

    logits_labels = []   # rep_block, subject, window block

    for EMG,GLOVE,ACC in loader:
        EMG=EMG.to("cuda").to(torch.float32).squeeze(0)
        GLOVE=GLOVE.to("cuda").to(torch.float32).squeeze(0)
        ACC=ACC.to("cuda").to(torch.float32).squeeze(0)
        cnt=total//EMG.shape[0]
        with torch.no_grad():
            logits = model.forward(EMG,ACC,GLOVE,total_tasks,total_tasks)

            # take mean for all 15 ms windows
            #logits_save = logits.detach().cpu().numpy()
            #for j, log in enumerate(logits):
            #    np.save('../data/logits_%d_%d.npy' % (i, j), logits_save)

            #logits_labels.append(dataset.get_idx_(i))

            loss=model.loss(logits)
            total_loss.append(loss.item())

            correct=model.correct_glove(logits.detach())

            total_correct.append(correct)
            total+=EMG.shape[0]

    total_loss=np.array(total_loss)
    #logits_labels=np.array(logits_labels)
    #np.save('../data/logits_labels.npy', logits_labels)
    print("Testing finished, logits saved, max %d"%(len(dataset)-1))

    return total_loss.mean(), sum(total_correct)/total


def validate(model, dataset):
    dataset.set_val()
    model.set_val()
    model.eval()

    total_tasks=dataset.TASKS
    total_loss = []
    total_correct = []
    total=0
    print("Validation size:", len(dataset))
    idx_val = torch.randperm(len(dataset))
    loader=data.DataLoader(dataset, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=PREFETCH)

    for EMG,GLOVE,ACC in loader:
        EMG=EMG.to("cuda").to(torch.float32).squeeze(0)
        GLOVE=GLOVE.to("cuda").to(torch.float32).squeeze(0)
        ACC=ACC.to("cuda").to(torch.float32).squeeze(0)
        with torch.no_grad():
            logits = model.forward(EMG,ACC,GLOVE,total_tasks,total_tasks)
            loss=model.loss(logits)
            total_loss.append(loss.item())
            correct=model.correct_glove(logits)

            total_correct.append(correct)
            total+=EMG.shape[0]

    total_loss=np.array(total_loss)
    return total_loss.mean(), sum(total_correct)/total

def train_loop(dataset, params, checkpoint=False, checkpoint_dir="../checkpoints/model", annealing=False,load=None):

    # cross validation parameters
    model = Model(d_e=params['d_e'], train_model=True, device="cuda").to(torch.float32)
    if load is not None:
        print("Loading model")
        model.load_state_dict(torch.load(checkpoint_dir+"%d.pth"%load))
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=0)
    if annealing:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0,verbose=True)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10**4,gamma=1, verbose=True)

    # train data
    dataset.set_train()
    total_tasks = dataset.TASKS
    idx_train = torch.randperm(len(dataset))

    loader=data.DataLoader(dataset, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=PREFETCH)

    val_losses={}
    counter=0

    print("Training...")
    for e in tqdm.trange(params['epochs']):
        loss_train=[]
        correct_amt=0
        total=0
        
        model.set_train()
        model.train()
        dataset.set_train()

        for EMG,GLOVE,ACC in loader:
            EMG=EMG.to("cuda").to(torch.float32).squeeze(0)
            GLOVE=GLOVE.to("cuda").to(torch.float32).squeeze(0)
            ACC=ACC.to("cuda").to(torch.float32).squeeze(0)

            #with amp.autocast():
            loss,logits=model.forward(EMG,ACC,GLOVE,total_tasks,total_tasks)
            l2=model.l2()*params['l2']
            loss=loss+l2

            for p in model.parameters():
                p.grad = None
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                loss_train.append((loss-l2).item())
                correct=model.correct_glove(logits.detach())
                correct_amt+=correct
                total+=EMG.shape[0]

        scheduler.step()
       
        acc_train=(correct_amt/total)
        loss_val,acc=validate(model, dataset)
        loss_train=np.array(loss_train).mean()
        print("Epoch %d. Train loss: %.4f\tVal loss: %.4f\tVal acc: %.6f\tTrain acc: %.4f" % (e, loss_train, loss_val, acc, acc_train))

        final_val_acc=(loss_val,acc)
        val_losses[e]=loss_val

        if checkpoint and loss_val <= max(list(val_losses.values())):
            print("Checkpointing model...")
            torch.save(model.state_dict(), checkpoint_dir+"%d.pt"%counter)
            counter+=1

    return final_val_acc, model


def cross_validate(lrs, regs, des, dataset, epochs=6, save=True):
    print(lrs)
    print(regs)
    print(des)
    cross_val={}

    for d_e in des:
        for i in range(len(lrs)):
            lr, reg = lrs[i], regs[i]
            print("d_e: %s, lr: %s, reg: %s"%(str(d_e),str(lr),reg))
            params = {
                    'd_e' : d_e,
                    'epochs' : epochs,
                    'lr' : lr,
                    'l2' : reg
                    }

            (loss_t,acc_t),model=train_loop(dataset, params, checkpoint=True)
            cross_val[(int(d_e), lr, reg)]=(loss_t,acc_t) #, loss_t) #TODO

    if save:
        values = np.array(list(cross_val.values()))
        keys = np.array(list(cross_val.keys()))
        np.save("../data/cross_val_values.npy", values)
        np.save("../data/cross_val_keys.npy", keys)

    return cross_val

def main():
    # DATASET - this is just for loading, change at will (no need to resample)
    new_people=3
    new_tasks=4

    dataset23 = DB23(new_tasks=new_tasks,new_people=new_people, device="cpu")

    params = {
            'd_e' : 128,
            'epochs' : 2,
            'lr' : 1e-2,
            'l2' : 1e-8
            }

    t=time.time()
    final_vals, model = train_loop(dataset23, params, checkpoint=True,annealing=True, checkpoint_dir="../checkpoints/testing")
    print(time.time()-t, time.time()-time.time())
    #final_stats=test(model, dataset23)
    #print("Final validation model statistics")
    #print(final_vals)
    #print("loss,\t\t\tcorrect")
    #print(final_stats)

    """
    # No train loader for now. Let's see performance.
    #train_loader=data.DataLoader(dataset=dataset23,batch_size=1,shuffle=True,num_workers=NUM_WORKERS)

    lrs = 10**np.random.uniform(low=-7, high=0, size=(30,))
    regs = 10**np.random.uniform(low=-9, high=2, size=(19,))
    regs = list(regs) + [0.0]*11
    des=[64,128]
    epochs=6
    cross_val = cross_validate(lrs, regs, des, dataset23, epochs=epochs, save=True)
    pprint(cross_val)

    # get best
    vals = np.array(list(cross_val.values()))
    best_val = vals[:, 1].argmax()
    keys = np.array(list(cross_val.keys()))
    best_key = keys[best_val]
    print("Best combination: %s" % str(best_key))
    print(vals[:, 1].sort())

    # test model
    d_e, lr, reg = best_key     # best model during validation

    print("Final training of model")
    params = {
            'd_e' : int(d_e),
            'epochs' : 200,
            'lr' : lr,
            'l2' : reg
            }

    final_vals, model = train_loop(dataset23, params, checkpoint=True,annealing=True, checkpoint_dir="../checkpoints/model_v3")
    final_stats=test(model, dataset23)

    print("Final validation model statistics")
    print(final_vals)
    print("loss,\t\t\tcorrect")
    print(final_stats)
    """

if __name__=="__main__":
    main()
