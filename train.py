# ************************************************************
# Author : Bumsoo Kim, 2018
# Github : https://github.com/meliketoy/graph-cnn.pytorch
#
# Korea University, Data-Mining Lab
# Graph Convolutional Neural Network
#
# Description : train.py
# The main code for training classification networks.
# ***********************************************************

import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from models import GCN

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
adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset="pubmed")

use_gpu = torch.cuda.is_available()

print("\n[STEP 2] : Obtain (adjacency, feature, label) matrix")
print("| Adjacency matrix : {}".format(adj.shape))
print("| Feature matrix   : {}".format(features.shape))
print("| Label matrix     : {}".format(labels.shape))

model = GCN(
        nfeat = features.shape[1],
        nhid = 1024,
        nclass = labels.max().item() + 1,
        dropout = 0.5
)

optimizer = optim.SGD(
        model.parameters(),
        lr = 1e-3,
        weight_decay = 0,
        momentum = 0.9
)

if use_gpu:
    _, features, adj, labels, idx_train, idx_val, idx_test = \
            list(map(lambda x: x.cuda(), [model, features, adj, labels, idx_train, idx_val, idx_test]))

print("\n[STEP 3] : Dummy Forward")
output = model(features, adj)
print("| Shape of result : {}".format(output.shape))

best_model = model
best_acc = 0

def train(epoch):
    global best_model
    global best_acc

    if (epoch % 500 == 0):
        print("=> Training Epoch #{} : lr = {}".format(epoch, 1e-2))
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    if (epoch % 500 == 0):
        print("| Training acc : {}%".format(acc_train.data.cpu().numpy() * 100))

    loss_train.backward()
    optimizer.step()

    model.eval()
    output = model(features, adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    if (epoch % 500 == 0):
        print("| Validation acc : {}%\n".format(acc_val.data.cpu().numpy() * 100))

    if acc_val > best_acc:
        print("=> Training Epoch #{} : lr = {}".format(epoch, 1e-2))
        print("| Training acc : {}%".format(acc_train.data.cpu().numpy() * 100))
        print("| Best acc : {}%". format(acc_val.data.cpu().numpy() * 100))
        best_acc = acc_val
        best_model = model

def test():
    print("\n[STEP 4] : Testing")
    best_model.eval()
    output = best_model(features, adj)
    acc_val = accuracy(output[idx_test], labels[idx_test])

    print("| Test acc : {}%\n".format(acc_val.data.cpu().numpy() * 100))

if __name__ == "__main__":
    print("\n[STEP 4] : Training")
    for epoch in range(10000):
        train(epoch)

    test()
