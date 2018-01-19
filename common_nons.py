import numpy as np
import scipy.sparse as ss
import tensorflow as tf

import scipy.sparse as ss
# from scipy.linalg import svd
# from scipy.sparse.linalg import svds

def get_adj_pmi(tokenized_docs, vocab_size):
    for tokenized_doc in tokenized:
        adj = np.vstack((tokenized_doc[:-1], tokenized_doc[1:]))

        # uni probs
        uni_index, uni_count = np.unique(tokenized_doc, return_counts=True)
        uni_prob = dict(zip(*(uni_index, uni_count / sum(uni_count))))

        # symmetric bi(gram) probs
        bi_index, bi_count = np.unique(np.sort(adj.T), return_counts=True, axis=0)
        bi_prob = zip(*(bi_index, bi_count / sum(bi_count)))

        # Pointwise mutual information
        pmi = np.zeros((bi_count.shape))
        for i, ((x, y), pxy) in enumerate(bi_prob):
            pxpy = uni_prob[x] * uni_prob[y]
            pmi[i] = np.log(pxy / pxpy)

        # DEBUG: print top 10 co-occurences
        # for i, j in bi_index[np.argsort(pmi)[-10:]]:
        #     print(id2word[i], id2word[j])

        # Sparse pmi graph
        doc_gr = ss.coo_matrix((pmi, (bi_index[:, 0],
                                      bi_index[:, 1])),
                               (vocab_size, vocab_size))

        yield doc_gr

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
