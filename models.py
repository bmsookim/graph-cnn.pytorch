import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)

        self.gc_path_1_1 = GraphConvolution(nhid, 16)
        self.gc_path_1_2 = GraphConvolution(16, 16)
        self.gc_path_1_3 = GraphConvolution(16, nhid)

        self.gc_path_2_1 = GraphConvolution(nhid, 16)
        self.gc_path_2_2 = GraphConvolution(16, 16)
        self.gc_path_2_3 = GraphConvolution(16, nhid)

        self.gc_path_3_1 = GraphConvolution(nhid, 16)
        self.gc_path_3_2 = GraphConvolution(16, 16)
        self.gc_path_3_3 = GraphConvolution(16, nhid)

        self.gc_path_4_1 = GraphConvolution(nhid, 16)
        self.gc_path_4_2 = GraphConvolution(16, 16)
        self.gc_path_4_3 = GraphConvolution(16, nhid)

        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def bottleneck(self, path1, path2, path3, adj, in_x):
        return F.relu(path3(F.relu(path2(F.relu(path1(in_x, adj)), adj)), adj))

    def forward(self, x, adj):
        skip = x
        x = F.dropout(F.relu(self.gc1(x, adj)), self.dropout, training=self.training)
        x = x + F.relu(self.gc1(skip, adj))

        skip = x
        path1 = self.bottleneck(self.gc_path_1_1, self.gc_path_1_2, self.gc_path_1_3, adj, x)
        path2 = self.bottleneck(self.gc_path_2_1, self.gc_path_2_2, self.gc_path_2_3, adj, x)
        path3 = self.bottleneck(self.gc_path_3_1, self.gc_path_3_2, self.gc_path_3_3, adj, x)
        path4 = self.bottleneck(self.gc_path_4_1, self.gc_path_4_2, self.gc_path_4_3, adj, x)
        NEXT = path1+path2+path3+path4

        x = NEXT + skip
        x = self.gc2(x, adj)

        return F.log_softmax(x, dim=0)
