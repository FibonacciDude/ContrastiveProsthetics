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
import sklearn.metrics as me

torch.cuda.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.benchmark=True
#torch.backends.cudnn.deterministic=True
shuff=True

def test(model, dataset, save="../data/"):
    dataset.set_test()
    model.set_test()

    total_loss = []
    loader=data.DataLoader(dataset, batch_size=args.batch_size, shuffle=shuff)
    logs = []

    for (EMG, GLOVE, label) in loader:
        label=label.reshape(-1)
        with torch.no_grad():
            #with amp.autocast():
            # logits of shape batch_size x (41 x 41)
            logits=model.forward(EMG, GLOVE, label)
            loss=model.loss(logits, label)
            total_loss.append(loss.item())
            logs.append(logits)

    # this is the RAW logits (for logits)
    logs = torch.cat(logs).detach().cpu().numpy()
    np.save(save+"logs.npy", logs)
    acc=model.correct()
    mean_loss=np.array(total_loss).mean()

    # get other things - this is in the 250 ms window
    # this is RAW predictions
    y_pred = model.y_pred_raw().flatten()
    y_true = model.y_true_raw().flatten()
    np.save(save+"y_pred.npy", y_pred)
    np.save(save+"y_true.npy", y_true)

    # voting window
    voting = model.voting_raw()
    np.save(save+"voting.npy", voting)

    # confusion matrix
    confusion_matrix = me.confusion_matrix(y_true, y_pred)
    np.save(save+"voting.npy", voting)
    print(confusion_matrix)

    return mean_loss, acc

def main(args):
    dataset23 = DB23(db2=args.db2)
    print("Loading dataset")
    dataset23.load_stored()
    print("Dataset loaded")
    dataset23=TaskWrapper(dataset23)

    model = Model(params=params, train_model=True, adabn=args.no_adabn, prediction=args.prediction, glove=args.glove, device="cuda").to(torch.float32)

    checkpoint_dir = "../checkpoints/contrastive"
    model.load_state_dict(torch.load(checkpoint_dir+".pt"))

    final_stats=test(model, dataset23)
    print("loss,\t\t\tcorrect")
    print(final_stats)

        
    # if all these change, the train set must change too (see how to edit such that the dataset includes all train)

    # solve - data info class or something (where???)
        # simple thing where you give it the model and the raw logits (stored...somewhere) - if we have
        # the logits, everything else is quite simple

        # add features of taking weird average and stuff between these models (in some way)

    # one crossvalidation -> use approximate values for training all models (might not be optimal...)

    # a few things to vary (if we want to): (args 1,2,3,4...)
        # rep split
        # subjects
        # tasks

    # show values:
        # per class accuracy (mean, std)
        # unbiased micro
        # biased macro (mean, std)
        # per person accuracy (and get std)
        # precision, recall, f1, etc...
        # curve of time taken vs accuracy of voting scheme prediction
    # get values:
        # time taken for forward pass
        # mean and std of the dataset (for all datasets) -> this will get overridden if we have multiple of them...

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Training on ninapro dataset')
    parser.add_argument('--crossval_size', type=int, default=10)
    parser.add_argument('--crossval_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--final_epochs', type=int, default=10)
    parser.add_argument('--glove', action='store_true')
    parser.add_argument('--db2', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--crossval_load', action='store_true')
    parser.add_argument('--prediction', action='store_true')
    parser.add_argument('--no_adabn', action='store_false')
    parser.add_argument('--no_checkpoint', action='store_false')
    parser.add_argument('--no_verbose', action='store_false')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    main(args)
