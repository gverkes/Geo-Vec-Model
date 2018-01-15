import numpy as np
import scipy.sparse as ss
import tensorflow as tf

def get_adj(tokenized_docs, vocab_size):
    for tokenized_doc in tokenized_docs:
        adj = np.vstack((tokenized_doc[:-1], tokenized_doc[1:]))
        sp_adj = ss.coo_matrix((np.ones(adj.shape[1]), (adj[0, :], adj[1, :])),
                               (vocab_size, vocab_size))
        sp_adj.sum_duplicates()

        yield sp_adj

def get_lapl(tokenized_docs, vocab_size, renorm_trick=True):
    for adj in get_adj(tokenized_docs, vocab_size):
        if renorm_trick == True:
            _adj = adj + ss.eye(adj.shape[0])
        D_inv_sqrt = ss.diags(np.power(np.array(_adj.sum(1)), -0.5).flatten())
        L = _adj.dot(D_inv_sqrt).transpose().dot(D_inv_sqrt).tocoo()

        yield adj, L

def sparse_to_tuple(sparse_mx):
    if not ss.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def sp2tf(sp_t, shape=None):
    t = sparse_to_tuple(sp_t)

    if shape is not None:
        t[2] = shape
    tensor = tf.SparseTensorValue(t[0],t[1].astype(np.float32),t[2])
    return tensor
