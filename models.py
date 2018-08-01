import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, GraphAttention

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, init):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, init=init)
        self.gc2 = GraphConvolution(nhid, nclass, init=init)
        self.dropout = dropout

    def bottleneck(self, path1, path2, path3, adj, in_x):
        return F.relu(path3(F.relu(path2(F.relu(path1(in_x, adj)), adj)), adj))

    def forward(self, x, adj):
        x = F.dropout(F.relu(self.gc1(x, adj)), self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return F.log_softmax(x, dim=1)

class GCN_drop_in(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, init):
        super(GCN_drop_in, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, init=init)
        self.gc2 = GraphConvolution(nhid, nclass, init=init)
        self.dropout = dropout

    def bottleneck(self, path1, path2, path3, adj, in_x):
        return F.relu(path3(F.relu(path2(F.relu(path1(in_x, adj)), adj)), adj))

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.dropout(F.relu(self.gc1(x, adj)), self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return F.log_softmax(x, dim=1)

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttention(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttention(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
