import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch

def parse_index_file(filename):
    index = []

    for line in open(filename):
        index.append(int(line.strip()))

    return index

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1

    return np.array(mask, dtype=np.bool)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_data(path="/home/bumsoo/Data/gcn/", dataset="cora"):
    """
    ind.[:dataset].x     => the feature vectors of the training instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].y     => the one-hot labels of the labeled training instances (numpy.ndarray)
    ind.[:dataset].allx  => the feature vectors of both labeled and unlabeled training instances (csr_matrix)
    ind.[:dataset].ally  => the labels for instances in ind.dataset_str.allx (numpy.ndarray)
    ind.[:dataset].graph => the dict in the format {index: [index of neighbor nodes]} (collections.defaultdict)

    ind.[:dataset].tx => the feature vectors of the test instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].ty => the one-hot labels of the test instances (numpy.ndarray)

    ind.[:dataset].test.index => indices of test instances in graph, for the inductive setting
    """

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []

    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(path, dataset, names[i]), 'rb') as f:
            objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(path, dataset))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    features = torch.FloatTensor(np.array(features.todense()))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    idx_test = test_idx_range.tolist()

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)

    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test
