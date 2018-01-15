import numpy as np
import scipy.sparse as ss
import tensorflow as tf

def get_adj(tokenized_docs, vocab_size):
    for docidx in tokenized_docs:
        adj_i = np.vstack((docidx[:-1], docidx[1:]))
        adj_o = np.flip(adj_i, axis=0)
        sp_adj_i = ss.coo_matrix((np.ones(adj_i.shape[1]), (adj_i[0, :],
                                                            adj_i[1, :])),
                                 (vocab_size, vocab_size))
        sp_adj_o = ss.coo_matrix((np.ones(adj_o.shape[1]), (adj_o[0, :],
                                                            adj_o[1, :])),
                                 (vocab_size, vocab_size))

        yield docidx, sp_adj_o, sp_adj_i

def get_lapl(tokenized_docs, vocab_size, renorm_trick=True):
    for docidx, A_o, A_i in get_adj(tokenized_docs, vocab_size):
        if renorm_trick == True:
            _A_i = A_i + ss.eye(A_i.shape[0])
            _A_o = A_o + ss.eye(A_o.shape[0])
        D_inv_sqrt_i = ss.diags(np.power(np.array(_A_i.sum(1)), -0.5).flatten())
        D_inv_sqrt_o = ss.diags(np.power(np.array(_A_o.sum(1)), -0.5).flatten())
        L_i = _A_i.dot(D_inv_sqrt_i).transpose().dot(D_inv_sqrt_i).tocoo()
        L_o = _A_o.dot(D_inv_sqrt_o).transpose().dot(D_inv_sqrt_o).tocoo()

        yield docidx, A_o, A_i, L_o, L_i

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
