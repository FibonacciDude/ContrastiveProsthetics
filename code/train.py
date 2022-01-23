import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.cuda.amp as amp
import numpy as np
from load import DB23
from constants import *
from models import Model
from pprint import pprint
from time import time
from utils import TaskWrapper
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from torchsummary import summary

import line_profiler, builtins, atexit
profile=line_profiler.LineProfiler()
atexit.register(profile.print_stats)

torch.cuda.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.determinist=True
shuff=True

def test(model, dataset):
    dataset.set_test()
    model.set_test()

    total_loss = []
    loader=data.DataLoader(dataset, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH, batch_size=args.batch_size, shuffle=shuff)

    for (EMG, GLOVE, label) in loader:
        label=label.reshape(-1)
        with torch.no_grad():
            #with amp.autocast():
            logits=model.forward(EMG, GLOVE)
            loss=model.loss(logits, label)
            total_loss.append(loss.item())

    acc=model.correct()
    mean_loss=np.array(total_loss).mean()
    return mean_loss, acc

def validate(model, dataset):
    dataset.set_val()
    model.set_val()

    total_loss = []
    loader=data.DataLoader(dataset, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH, batch_size=args.batch_size, shuffle=shuff)

    for (EMG, GLOVE, label) in tqdm(loader):
        label=label.reshape(-1)
        with torch.no_grad():
            #with amp.autocast():
            logits=model.forward(EMG, GLOVE)
            loss=model.loss(logits, label)
            total_loss.append(loss.item())

    acc=model.correct()
    mean_loss=np.array(total_loss).mean()
    return mean_loss, acc

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_loop(dataset, params, checkpoint=False, checkpoint_dir="../checkpoints/model", annealing=False, load=None, verbose=False):
    model = Model(params=params, train_model=True, adabn=args.no_adabn, prediction=args.prediction, glove=args.glove, device="cuda").to(torch.float32)
    print(count_parameters(model))

    if load is not None:
        print("Loading model")
        model.load_state_dict(torch.load(checkpoint_dir+"%d.pt"%load))

    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=0) # batchnorm wrong with AdamW
    if annealing:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0,verbose=True)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10**4, gamma=1, verbose=True)

    # train data
    dataset.set_train()
    model.set_train()

    loader=data.DataLoader(dataset, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH, batch_size=args.batch_size, shuffle=shuff)

    val_losses={}
    counter=0
   
    print("Training...")
    for e in trange(params['epochs']):
        
        loss_train=[]
        for (EMG, GLOVE, label) in tqdm(loader):
            label=label.reshape(-1)
            #with amp.autocast():
            logits=model.forward(EMG, GLOVE)
            loss=model.loss(logits, label)
            loss_train.append(loss.item())
            loss=loss+model.l2()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        acc_train=model.correct()

        scheduler.step()
       
        if verbose:
            loss_val,acc_val=validate(model, dataset)
            loss_train=np.array(loss_train).mean()
            final_val_acc=(loss_val,acc_val)
            val_losses[e]=loss_val
            print("Epoch %d. Train loss: %.4f\tVal loss: %.4f\tVal acc: %.6f\tTrain acc: %.4f" % (e, loss_train, loss_val, acc_val, acc_train))

        if checkpoint and loss_val <= max(list(val_losses.values())):
            print("Checkpointing model...")
            torch.save(model.state_dict(), checkpoint_dir+"%d.pt"%counter)
            counter+=1

        model.set_train()
        dataset.set_train()


    if not verbose:
        loss_val,acc_val=validate(model, dataset)
        loss_train=np.array(loss_train).mean()
        print("Epoch %d. Train loss: %.4f\tVal loss: %.4f\tVal acc: %.6f\tTrain acc: %.4f" % (e, loss_train, loss_val, acc_val, acc_train))
        final_val_acc=(loss_val,acc_val)
        val_losses[e]=loss_val

    return final_val_acc, model

def cross_validate(des, hyperparams, dataset, epochs=6, save=True, load=False):
    cross_val={}
    if not load:
        for d_e in des:
            for hypervals in zip(*list(hyperparams.values())):
                current = {
                    key : val for key,val in zip(list(hyperparams.keys()), hypervals)
                    }
                print(current)
                params = {
                        'd_e' : d_e,
                        'epochs' : epochs,
                        }
                params.update(current)
                (loss_t,acc_t),model=train_loop(dataset, params, checkpoint=False, verbose=False)
                cross_val[(d_e,)+tuple(hypervals)]=(loss_t,acc_t)

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
    dataset23 = DB23()
    print("Loading dataset")
    dataset23.load_stored()
    print("Dataset loaded")
    dataset23=TaskWrapper(dataset23)

    lrs = 10**np.random.uniform(low=-6, high=0, size=(args.crossval_size,))
    regs_emg = 10**np.random.uniform(low=-9, high=1, size=(args.crossval_size,))
    dps_emg = np.random.uniform(low=0, high=.9, size=(args.crossval_size,))
    regs_glove = 10**np.random.uniform(low=-9, high=1, size=(args.crossval_size,))
    dps_glove = np.random.uniform(low=0, high=.9, size=(args.crossval_size,))
    des=[16]

    hyperparams = {
            'lr' : lrs,
            'reg_emg' : regs_emg,
            'dp_emg' : dps_emg,
            'reg_glove' : regs_glove,
            'dp_glove' : dps_glove,
            }

    values, keys = cross_validate(des, hyperparams, dataset23, epochs=args.crossval_epochs, save=True, load=args.crossval_load)

    # get best
    best_val = np.nanargmax(values[:, 1])
    best_key = keys[best_val]
    print("Best combination: %s" % str(best_key))

    # test model
    d_e, lr, reg_e, dp_e, reg_g, dp_g = best_key     # best model during validation

    params = {
            'd_e' : int(d_e),
            'epochs' : args.final_epochs,
            'lr' : lr,
            'dp_emg' : dp_e,
            'reg_emg' : reg_e,
            'dp_glove' : dp_g,
            'reg_glove' : reg_g
            }

    print("Final training of model")
    final_vals, model = train_loop(dataset23, params, checkpoint=args.no_checkpoint, annealing=True, checkpoint_dir="../checkpoints/model_v8", verbose=args.no_verbose)
    print("Final validation model statistics")
    print(final_vals)

    if args.test:
        # not until very very end
        final_stats=test(model, dataset23)
        print("loss,\t\t\tcorrect")
        print(final_stats)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Training on ninapro dataset')
    parser.add_argument('--crossval_size', type=int, default=100)
    parser.add_argument('--crossval_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--final_epochs', type=int, default=100)
    parser.add_argument('--glove', action='store_true')
    parser.add_argument('--crossval_load', action='store_true')
    parser.add_argument('--prediction', action='store_true')
    parser.add_argument('--no_adabn', action='store_false')
    parser.add_argument('--no_checkpoint', action='store_false')
    parser.add_argument('--no_verbose', action='store_false')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    main(args)
