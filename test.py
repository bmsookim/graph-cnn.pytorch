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
adj, features, labels, _, _, idx_test = load_data(dataset="pubmed")
use_gpu = torch.cuda.is_available()

print("\n[STEP 2] : Obtain (adjacency, feature, label) matrix")
print("| Adjacency matrix : {}".format(adj.shape))
print("| Feature matrix   : {}".format(features.shape))
print("| Label matrix     : {}".format(labels.shape))

if use_gpu:
    _, features, adj, labels, idx_train, idx_val, idx_test = \
            list(map(lambda x: x.cuda(), [model, features, adj, labels, idx_train, idx_val, idx_test]))

print("\n[STEP 3] : Dummy Forward")
output = model(features, adj)
print("| Shape of result : {}".format(output.shape))

def test():
    print("\n[STEP 4] : Testing")
    best_model = load_model() #load model implementeation required
    best_model.eval()
    output = best_model(features, adj)
    acc_val = accuracy(output[idx_test], labels[idx_test])
    print("| Test acc : {}%\n".format(acc_val.data.cpu().numpy() * 100))

if __name__ == "__main__":
    test()
