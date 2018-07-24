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
"""
adj, features, labels_train, labels_val, labels_test = load_data(dataset="cora")

use_gpu = torch.cuda.is_available()

print(adj.shape)
print(features.shape)

model = GCN(
        nfeat = features.shape[1],
        nhid = 16,
        nclass = labels_train.shape[1],
        dropout = 0.5
)

output = model(features, adj)
print(output.shape)
