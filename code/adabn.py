import torch 
import torch.nn as nn
from utils import RunningStats

class AdaBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=.1, affine=True):
        super(AdaBatchNorm2d, self).__init__()
        self.stats = []
        self.domain_labels = []
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats=False)

    def forward(self, X):
        if self.training:
            return self.bn(X)

        if self.domain not in self.domain_labels:
            self.domain_labels.append(self.domain)

            rs = RunningStats(mode=True)
            self.stats.append(rs)

        idx=self.domain_labels.index(self.domain)
        for x in X:
            self.stats[idx].push(x)
        mean,std=self.stats[idx].mean_std()
        X=(X-mean[None, :, None, None])/std[None, :, None, None]
        if self.bn.affine:
            X = X * self.bn.weight[None, :, None, None] + self.bn.bias[None, :, None, None]
        return X

class AdaBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=.1, affine=True):
        super(AdaBatchNorm1d, self).__init__()
        self.stats = []
        self.domain_labels = []
        self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats=False)

    def forward(self, X):
        if self.training:
            return self.bn(X)

        if self.domain not in self.domain_labels:
            self.domain_labels.append(self.domain)

            rs = RunningStats(mode=False)
            self.stats.append(rs)

        idx=self.domain_labels.index(self.domain)
        for x in X:
            self.stats[idx].push(x)
        mean,std=self.stats[idx].mean_std()
        X=(X-mean[None])/std[None]
        if self.bn.affine:
            X=X*self.bn.weight+self.bn.bias
        return X

