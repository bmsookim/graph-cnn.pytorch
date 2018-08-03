# ************************************************************
# Author : Bumsoo Kim, 2018
# Github : https://github.com/meliketoy/graph-cnn.pytorch
#
# Korea University, Data-Mining Lab
# Graph Convolutional Neural Network
#
# Description : test.py
# The main code for testing graph classification networks.
# ***********************************************************

import time
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from models import GCN
from opts import TestOptions

"""
N : number of nodes
D : number of features per node
E : number of classes

@ input :
    - adjacency matrix (N x N)
    - feature matrix (N x D)
    - label matrix (N x E)

@ dataset :
    - citeseer
    - cora
    - pubmed
"""
opt = TestOptions().parse()

adj, features, labels, idx_train, idx_val, idx_test = load_data(path=opt.dataroot, dataset=opt.dataset)
use_gpu = torch.cuda.is_available()

print("\n[STEP 2] : Obtain (adjacency, feature, label) matrix")
print("| Adjacency matrix : {}".format(adj.shape))
print("| Feature matrix   : {}".format(features.shape))
print("| Label matrix     : {}".format(labels.shape))

load_model = torch.load(os.path.join('checkpoint', opt.dataset, '%s.t7' %(opt.model)))
model = load_model['model'].cpu()
acc_val = load_model['acc']

if use_gpu:
    _, features, adj, labels, idx_test = \
            list(map(lambda x: x.cuda(), [model, features, adj, labels, idx_test]))

def test():
    print("\n[STEP 4] : Testing")

    model.eval()
    output = model(features, adj)

    print(output[idx_test].shape)
    print(labels[idx_test].shape)

    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("| Validation acc : {}%".format(acc_val.data.cpu().numpy() * 100))
    print("| Test acc : {}%\n".format(acc_test.data.cpu().numpy() * 100))

if __name__ == "__main__":
    test()
