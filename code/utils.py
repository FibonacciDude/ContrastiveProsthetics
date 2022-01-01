import numpy as np
import torch

# https://www.johndcook.com/blog/standard_deviation/

class RunningStats():
    def __init__(self):
        self.counter = 0

    def push(self, X):
        self.counter+=1
        X=X.mean(0) # along window size dimension (constant for all X)
        if self.counter==1:
            self.old_mean = self.new_mean = X
            self.old_std = X.clone() * 0.0
        else:
            self.new_mean=self.old_mean + (X-self.old_mean)/self.counter
            self.new_std=self.old_std+(X-self.old_mean)*(X-self.new_mean)
            self.old_mean=self.new_mean
            self.old_std=self.new_std

    def mean(self):
        return self.new_mean

    def variance(self):
        return (self.new_std)/(self.counter-1)

    def std(self):
        return torch.sqrt(self.variance())

    def mean_std(self):
        return self.mean(), self.std()
