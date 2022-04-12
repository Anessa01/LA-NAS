from ast import Constant
import numpy as np
import pickle as pkl
import copy
import sys
import torch

import logging
 


_opname_to_index = {
    'none': 0,
    'skip_connect': 1,
    'nor_conv_1x1': 2,
    'nor_conv_3x3': 3,
    'avg_pool_3x3': 4,
    'input': 5,
    'output': 6,
    'global': 7
}


def normalize_adj_simple(adj):
    #The model saved before 20220411 are using Dr^-1 A Dc^-1 instead of Dr^-(1/2) A Dc^-(1/2)
    rowsum = np.array(adj.sum(1))
    colsum = np.array(adj.sum(0))
    d_inv_sqrt_r = np.power(rowsum, -1.).flatten()
    d_inv_sqrt_r[np.isinf(d_inv_sqrt_r)] = 0.
    d_mat_inv_sqrt_r = np.diag(d_inv_sqrt_r)
    d_inv_sqrt_c = np.power(colsum, -1.).flatten()
    d_inv_sqrt_c[np.isinf(d_inv_sqrt_c)] = 0.
    d_mat_inv_sqrt_c = np.diag(d_inv_sqrt_c)
    return np.matmul(d_mat_inv_sqrt_r, adj, d_mat_inv_sqrt_c)

def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def get_matrix_and_ops(g, prune=True, keep_dims=False):
    ''' Return the adjacency matrix and label vector.

        Args:
            g : should be a point from Nasbench102 search space
            prune : remove dangling nodes that only connected to zero ops
            keep_dims : keep the original matrix size after pruning
    '''

    matrix = [[0 for _ in range(8)] for _ in range(8)]
    labels = [None for _ in range(8)]
    labels[0] = '5'
    labels[-1] = '6'
    matrix[0][1] = matrix[0][2] = matrix[0][4] = 1
    matrix[1][3] = matrix[1][5] = 1
    matrix[2][6] = 1
    matrix[3][6] = 1
    matrix[4][7] = 1
    matrix[5][7] = 1
    matrix[6][7] = 1

    for idx, op in enumerate(g):
        if op == 0: # zero
            for other in range(8):
                if matrix[other][idx+1]:
                    matrix[other][idx+1] = 0
                if matrix[idx+1][other]:
                    matrix[idx+1][other] = 0
        elif op == 1: # skip-connection:
            to_del = []
            for other in range(8):
                if matrix[other][idx+1]:
                    for other2 in range(8):
                        if matrix[idx+1][other2]:
                            matrix[other][other2] = 1
                            matrix[other][idx+1] = 0
                            to_del.append(other2)
            for d in to_del:
                matrix[idx+1][d] = 0
        else:
            labels[idx+1] = str(op)
        
    if prune:
        visited_fw = [False for _ in range(8)]
        visited_bw = copy.copy(visited_fw)

        def bfs(beg, vis, con_f):
            q = [beg]
            vis[beg] = True
            while q:
                v = q.pop()
                for other in range(8):
                    if not vis[other] and con_f(v, other):
                        q.append(other)
                        vis[other] = True
                
        bfs(0, visited_fw, lambda src, dst: matrix[src][dst]) # forward
        bfs(7, visited_bw, lambda src, dst: matrix[dst][src]) # backward
    
        for v in range(7, -1, -1):
            if not visited_fw[v] or not visited_bw[v]:
                labels[v] = None
                if keep_dims:
                    matrix[v] = [0] * 8
                else:
                    del matrix[v]
                for other in range(len(matrix)):
                    if keep_dims:
                        matrix[other][v] = 0
                    else:
                        del matrix[other][v]
    
        if not keep_dims:
            labels = list(filter(lambda l: l is not None, labels))
            
        assert visited_fw[-1] == visited_bw[0]
        assert visited_fw[-1] == False or matrix
    
        verts = len(matrix)
        assert verts == len(labels)
        for row in matrix:
            assert len(row) == verts
    
    return matrix, labels



def add_global(k):
    adj = [0]
    for i in range(len(k)):
        adj.append(1)
        k[i].insert(0, 0)
    k.insert(0, adj)
    return k

def load_data(dataset_str = 'data/desktop-cpu-core-i7-7820x-fp32.pickle', sep = 0.9):
    f = open(dataset_str, 'rb')
    m = pkl.load(f, encoding='latin1')
    total_len = len(m)
    graph = list(m.keys())
    y = list(m.values())

    train_len = int(sep * total_len)
    test_len = total_len - train_len
    train_x = graph[:train_len]
    test_x = graph[train_len:]
    train_y = y[:train_len]
    test_y = y[train_len:]

    adj = []
    feature = []
    for i in range(train_len):
        adtemp, fetemp = get_matrix_and_ops(train_x[i])
        adtemp = np.array(add_global(adtemp))
        adtemp = normalize_adj_simple(adtemp)
        adj.append(adtemp)
        fetemp = [(int(i) - 2)  for i in fetemp]
        fetemp.insert(0, 6)
        fetemp = np.array(fetemp)
        feature.append(np.eye(7)[fetemp])
    adj_t = []
    feature_t = []
    for i in range(test_len):
        adtemp, fetemp = get_matrix_and_ops(test_x[i])
        adtemp = np.array(add_global(adtemp))
        adtemp = normalize_adj_simple(adtemp)
        adj_t.append(adtemp)
        fetemp = [(int(i) - 2)  for i in fetemp]
        fetemp.insert(0, 6)
        fetemp = np.array(fetemp)
        feature_t.append(np.eye(7)[fetemp])
    
    return adj, feature, train_y, adj_t, feature_t, test_y

def load_data_b(dataset_str = 'data/desktop-cpu-core-i7-7820x-fp32.pickle', sep = 0.9):
    f = open(dataset_str, 'rb')
    m = pkl.load(f, encoding='latin1')
    total_len = len(m)
    graph = list(m.keys())
    y = list(m.values())

    train_len = int(sep * total_len)
    test_len = total_len - train_len
    train_x = graph[:train_len]
    test_x = graph[train_len:]
    train_y = y[:train_len]
    test_y = y[train_len:]

    # train set
    adj = []
    feature = []
    for i in range(train_len):
        adtemp, fetemp = get_matrix_and_ops(train_x[i])
        adtemp = np.array(add_global(adtemp))
        adtemp = normalize_adj_simple(adtemp)
        L = 9 - len(adtemp)
        adtemp = np.pad(adtemp, ((0, L), (0, L)), 'constant')
        adj.append(adtemp)
        fetemp = [(int(i) - 2)  for i in fetemp]
        fetemp.insert(0, 6)
        fetemp = np.array(fetemp)
        fetemp = np.eye(7)[fetemp]
        fetemp = np.pad(fetemp, ((0, L),(0, 0)), 'constant')
        feature.append(fetemp)
    
    #test set
    adj_t = []
    feature_t = []
    for i in range(test_len):
        adtemp, fetemp = get_matrix_and_ops(test_x[i])
        adtemp = np.array(add_global(adtemp))
        adtemp = normalize_adj_simple(adtemp)
        L = 9 - len(adtemp)
        adtemp = np.pad(adtemp, ((0, L), (0, L)), 'constant')
        adj_t.append(adtemp)
        fetemp = [(int(i) - 2)  for i in fetemp]
        fetemp.insert(0, 6)
        fetemp = np.array(fetemp)
        fetemp = np.eye(7)[fetemp]
        fetemp = np.pad(fetemp, ((0, L), (0, 0)), 'constant')
        feature_t.append(fetemp)
    
    return adj, feature, train_y, adj_t, feature_t, test_y


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger




