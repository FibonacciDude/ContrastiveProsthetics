import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.cuda.amp as amp
import numpy as np
from load import DB23
from constants import *
from utils import Loader
from models import GLOVENet, EMGNet, Model
from pprint import pprint
import json, time, tqdm
import matplotlib.pyplot as plt

torch.manual_seed(42)
torch.backends.cudnn.benchmark=True
shuff=True

# test incomplete (not updated)
def test(model, dataset):
    dataset.set_test()
    model.set_test()
    model.eval()

    idx_test = torch.randperm(len(dataset))
    #loader=data.DataLoader(dataset, shuffle=shuff, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH)
    loader=Loader(dataset=dataset, batch_size=args.batch_size, grouped=args.no_grouped)
    print("Test size:", len(dataset))

    total_tasks=dataset.TASKS
    total_loss = []
    total_correct = []  # in each 15 ms
    total=0

    logits_labels = []   # rep_block, subject, window block

    for EMG,GLOVE,ACC in loader:
        EMG=EMG.to(torch.float32)
        GLOVE=GLOVE.to(torch.float32)
        ACC=ACC.to(torch.float32)

        if args.adabn:
            EMG=EMG.squeeze(0)
            GLOVE=GLOVE.squeeze(0)
            ACC=ACC.squeeze(0)

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
    idx_val = torch.randperm(len(dataset))
    #loader=data.DataLoader(dataset, shuffle=shuff, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH)
    loader=Loader(dataset=dataset, batch_size=args.batch_size, grouped=args.no_grouped)

    for EMG,GLOVE,ACC in loader:
        EMG=EMG.to(torch.float32)
        GLOVE=GLOVE.to(torch.float32)
        ACC=ACC.to(torch.float32)

        if args.adabn:
            EMG=EMG.squeeze(0)
            GLOVE=GLOVE.squeeze(0)
            ACC=ACC.squeeze(0)
        with torch.no_grad():
            loss = model.forward(EMG,ACC,GLOVE,dataset.tasks_mask)
            total_loss.append(loss.item())

    acc=model.correct_glove()
    total_loss=np.array(total_loss)
    return total_loss.mean(), acc

def train_loop(dataset, params, checkpoint=False, checkpoint_dir="../checkpoints/model", annealing=False, load=None, verbose=False):

    global args
    # cross validation parameters
    model = Model(d_e=params['d_e'], dp=params['dp'], adabn=args.adabn, train_model=True, device="cuda").to(torch.float32)

    if load is not None:
        print("Loading model")
        model.load_state_dict(torch.load(checkpoint_dir+"%d.pth"%load))

    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=0) # batchnorm wrong with AdamW
    if annealing:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0,verbose=True)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10**4, gamma=1, verbose=True)

    #scheduler = optim.lr_scheduler.OneCycleLR(optimizer, total_steps=len(dataset)*params['epochs'], max_lr=params['lr'], div_factor=3, final_div_factor=3, verbose=False) # super-convergence scheduler
    #scheduler = optim.lr_scheduler.CyclicLR(optimizer,base_lr=0, max_lr=2e-1, step_size_up=params['epochs']*len(dataset), step_size_down=0, verbose=False, cycle_momentum=False)

    # train data
    dataset.set_train()
    model.set_train()
    model.train()
    total_tasks = dataset.TASKS
    idx_train = torch.randperm(len(dataset))

    #loader=data.DataLoader(dataset, shuffle=shuff, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH)
    loader=Loader(dataset=dataset, batch_size=args.batch_size, grouped=args.no_grouped)

    val_losses={}
    counter=0
   
    #TODO: remove this later
    #lrs, running_acc = [], []

    print("Training...")
    for e in tqdm.trange(params['epochs']):
        
        model.set_train()
        model.train()
        dataset.set_train()

        loss_train=[]
        for EMG,GLOVE,ACC in loader:
            EMG=EMG.to(torch.float32)
            GLOVE=GLOVE.to(torch.float32)
            ACC=ACC.to(torch.float32)

            if args.adabn:
                EMG=EMG.squeeze(0)
                GLOVE=GLOVE.squeeze(0)
                ACC=ACC.squeeze(0)

            #with amp.autocast():
            loss=model.forward(EMG,ACC,GLOVE,dataset.tasks_mask)
            l2=model.l2()*params['l2']
            b_loss=(loss+l2)

            optimizer.zero_grad(set_to_none=True)
            b_loss.backward()
            optimizer.step()

            loss_train.append(loss.item())

        acc_train=model.correct_glove()

        """
        if e == 16-1 or e == 24-1:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / 10
            print("changing lr", g['lr'])
        """
        scheduler.step()
       
        if verbose:
            loss_val,acc_val=validate(model, dataset)
            loss_train=np.array(loss_train).mean()
            print("Epoch %d. Train loss: %.4f\tVal loss: %.4f\tVal acc: %.6f\tTrain acc: %.4f" % (e, loss_train, loss_val, acc_val, acc_train))
            final_val_acc=(loss_val,acc_val)
            val_losses[e]=loss_val

        if checkpoint and loss_val <= max(list(val_losses.values())):
            print("Checkpointing model...")
            torch.save(model.state_dict(), checkpoint_dir+"%d.pt"%counter)
            counter+=1

    if not verbose:
        loss_val,acc_val=validate(model, dataset)
        loss_train=np.array(loss_train).mean()
        print("Epoch %d. Train loss: %.4f\tVal loss: %.4f\tVal acc: %.6f\tTrain acc: %.4f" % (e, loss_train, loss_val, acc_val, acc_train))
        final_val_acc=(loss_val,acc_val)
        val_losses[e]=loss_val

        model.set_train()
        model.train()
        dataset.set_train()

    """
    plt.plot(lrs, running_acc)
    plt.show()
    best_lr=max(running_acc)
    print("Best lr found:", best_lr)
    """

    return final_val_acc, model


def cross_validate(lrs, regs, des, dps, dataset, epochs=6, save=True, load=False):
    cross_val={}
    if not load:
        for d_e in des:
            for i in range(len(lrs)):
                lr, reg, dp = lrs[i], regs[i], dps[i]
                print("d_e: %s, lr: %s, reg: %s, dp: %s"%(str(d_e),str(lr),reg, str(dp)))
                params = {
                        'd_e' : d_e,
                        'epochs' : epochs,
                        'lr' : lr,
                        'l2' : reg,
                        'dp' : dp,
                        }
                (loss_t,acc_t),model=train_loop(dataset, params, checkpoint=False, verbose=False)
                cross_val[(int(d_e), lr, reg, dp)]=(loss_t,acc_t) #, loss_t) #TODO

        values = np.array(list(cross_val.values()))
        keys = np.array(list(cross_val.keys()))
        if save:
            np.save("../data/cross_val_values.npy", values)
            np.save("../data/cross_val_keys.npy", keys)
    else:
        values=np.load("../data/cross_val_values.npy")
        keys=np.load("../data/cross_val_keys.npy")

    return values, keys

def main(args):
    # DATASET - this is just for loading, change at will (no need to resample)
    new_people=3
    new_tasks=4

    dataset23 = DB23(adabn=args.adabn)
    print("Loading dataset")
    dataset23.load_stored()
    print("Dataset loaded")

    lrs = [1e-4]*args.crossval_size
    regs = 10**np.random.uniform(low=-8, high=0, size=(args.crossval_size,))
    dps = np.random.uniform(low=.05, high=.95, size=(args.crossval_size,))
    #lrs = 10**np.random.uniform(low=-6, high=0, size=(args.crossval_size,))
    #regs = 10**np.random.uniform(low=-8, high=2, size=(args.crossval_size,))
    #regs = list(regs)
    #des=[64, 128, 256]
    des=[64]

    values, keys = cross_validate(lrs, regs, des, dps, dataset23, epochs=args.crossval_epochs, save=True, load=args.crossval_load)

    # get best
    best_val = values[:, 1].argmax()
    best_key = keys[best_val]
    print("Best combination: %s" % str(best_key))
    #print(sorted(list(values[:, 1])))

    # test model
    d_e, lr, reg = best_key     # best model during validation
    dp = 0.08319825261907951
    #d_e, lr, reg, dp = best_key     # best model during validation

    params = {
            'd_e' : int(d_e),
            'epochs' : args.final_epochs,
            'lr' : lr,
            'dp' : dp,
            'l2' : reg
            }
    print("Final training of model")
    final_vals, model = train_loop(dataset23, params, checkpoint=args.no_checkpoint, annealing=True, checkpoint_dir="../checkpoints/model_v7", verbose=args.no_verbose)
    print("Final validation model statistics")
    print(final_vals)

    # not until very very end
    #final_stats=test(model, dataset23)
    #print("loss,\t\t\tcorrect")
    #print(final_stats)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Training on ninapro dataset')
    parser.add_argument('--crossval_size', type=int, default=100)
    parser.add_argument('--crossval_epochs', type=int, default=1)
    parser.add_argument('--crossval_load', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--adabn', action='store_true')
    parser.add_argument('--final_epochs', type=int, default=100)
    parser.add_argument('--no_grouped', action='store_false')
    parser.add_argument('--no_checkpoint', action='store_false')
    parser.add_argument('--no_verbose', action='store_false')
    args = parser.parse_args()

    main(args)
