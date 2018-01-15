import numpy as np
import os
import scipy.sparse as ss
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

from common_nons import sp2tf, get_lapl


class GraphVec():
    def __init__(self, corpus=None, vocab_size=10, h_layers=[8, 4],
                 act=tf.nn.relu, learning_rate=1e-3,
                 pos_sample_size=512, embedding_size_w=128,
                 embedding_size_d=2, n_neg_samples=64,
                 window_size=8, window_batch_size=128, friendly_print=False):
        """Geo-Vec model as described in the report model section."""

        self.corpus = corpus
        self.corpus_size = len(corpus)
        self.vocab_size = vocab_size
        self.h_layers = h_layers
        self.act = act
        self.learning_rate = learning_rate
        self.pos_sample_size = pos_sample_size
        self.embedding_size_w = embedding_size_w
        self.embedding_size_d = embedding_size_d
        self.n_neg_samples = n_neg_samples
        self.window_size = window_size
        self.window_batch_size = window_batch_size

        # use for plotting
        self._loss_vals, self._acc_vals = [], []

        # placeholders
        # s = [self.vocab_size, self.vocab_size]
        self.placeholders = {
            'A': tf.sparse_placeholder(tf.float32),
            'L': tf.sparse_placeholder(tf.float32),
            'idx': tf.placeholder(tf.int64),
            'val': tf.placeholder(tf.float32),
            'train_dataset': tf.placeholder(tf.int32),
            'train_labels': tf.placeholder(tf.int32),
            # 'dropout': tf.placeholder_with_default(0., shape=())
        }

        # model
        # self.aux_losses = None
        dummy = sp2tf(ss.eye(self.vocab_size))
        self.init_model(x=dummy)
        self.samples = None
        self.current_sample = 0

        # saver
        self.saver = tf.train.Saver()

        # optimizer
        self.init_optimizer()

        # sess
        self.trained = 0
        # self.sess = tf.Session(graph=self.graph)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def init_model(self, x, aux_tasks=None):
        """geo-vec model with variable number of gcn layers. Optional aux_taks
        param is now unimplemented to specify which tasks to add. All aux losses
        should be gathered in a self.aux_losses variable to gather later on."""
        self.h = [self.gcn(x, self.vocab_size, self.h_layers[0], self.act, layer=0, sparse=True)]

        for i in range(1, len(self.h_layers)-1):
            self.h.append(self.gcn(self.h[-1], self.h_layers[i-1],
                                   self.h_layers[i], self.act, layer=i))

        self.emb = self.gcn(self.h[-1], self.h_layers[-2], self.h_layers[-1],
                            act=lambda x: x, layer=len(self.h_layers))

        # Auxiliary model
        self.word_embeddings = tf.Variable(
            tf.random_uniform([self.vocab_size, self.embedding_size_w], -1.0, 1.0))

        # concatenating word vectors and doc vector
        combined_embed_vector_length = self.embedding_size_w * self.window_size + \
            self.embedding_size_d * self.h_layers[-2]

        # softmax weights, W and D vectors should be concatenated before applying softmax
        self.weights = tf.Variable(
            tf.truncated_normal([self.vocab_size, combined_embed_vector_length],
                                stddev=1.0 / np.sqrt(combined_embed_vector_length)))

        # softmax biases
        self.biases = tf.Variable(tf.zeros([self.vocab_size]))

        # collect embedding matrices with shape=(batch_size, embedding_size)
        embed = []
        for j in range(self.window_size):
            embed_w = tf.nn.embedding_lookup(self.word_embeddings, self.placeholders['train_dataset'][:, j])
            embed.append(embed_w)

        self.doc_embeddings = tf.Variable(
             tf.random_uniform([self.embedding_size_d, self.vocab_size], -1.0, 1.0))

        # Save document embedding in original shape (i.e. B x D)
        self.embed_d = tf.matmul(self.doc_embeddings, self.h[-1])

        # Flatten document embedding to concatenate with word embeddings
        embed_d = tf.expand_dims(tf.reshape(self.embed_d, [-1]), 0)
        embed_d = tf.tile(embed_d, [tf.shape(embed[0])[0], 1])

        embed.append(embed_d)
        # concat word and doc vectors
        self.embed = tf.concat(embed, 1)

    def gcn(self, x, dim_in, dim_out, act, layer, sparse=False):
        """basic graph convolution using a split up adjacency matrix.
        The separation param is to create the final embeddings to reconstruct."""
        w = tf.get_variable('w_{}'.format(layer), shape=[dim_in, dim_out],
                             initializer=tf.contrib.layers.xavier_initializer())

        if sparse:
            x = tf.sparse_tensor_dense_matmul(x, w)
        else:
            x = tf.matmul(x, w)

        x = tf.sparse_tensor_dense_matmul(self.placeholders['L'], x)

        return self.act(x)

    def init_optimizer(self):
        """initializes optimizer and computes loss + accuracy. The loss function
        is currently a MSE, due to the fact we are dealing with weighted edges.
        This does not seem ideal, and should be thought about."""
        emb_rows = tf.gather(self.emb, self.placeholders['idx'][:, 0])
        emb_cols = tf.gather(self.emb, self.placeholders['idx'][:, 1])

        # Compute reconstruction values for sampled rows/columns
        self.recon = tf.reduce_sum(tf.multiply(emb_rows, emb_cols), 1)

        self.graph_loss = tf.losses.mean_squared_error(self.recon, self.placeholders['val'],
                                                       weights=self.get_weights(self.placeholders['val']))

        self.aux_loss = tf.nn.nce_loss(self.weights, self.biases, self.placeholders['train_labels'],
                                  self.embed, 128, self.vocab_size)
        self.aux_loss = tf.reduce_mean(self.aux_loss)

        self.loss = self.graph_loss + self.aux_loss

        # optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.opt_op = optimizer.minimize(self.loss)

        correct_recons = tf.greater_equal(tf.multiply(tf.cast(self.recon, tf.int32),
                                                      tf.cast(self.placeholders['val'], tf.int32)),
                                          tf.cast(self.placeholders['val'], tf.int32))

        self.accuracy = tf.reduce_mean(tf.cast(correct_recons, tf.float32))

    def get_weights(self, labels):
        """Compute positive and negative example weights"""
        pos_sum = tf.reduce_sum(labels)
        pos_weight = (64**2 - pos_sum) / pos_sum
        thresh = tf.fill(tf.shape(labels), 1.)
        x = tf.fill(tf.shape(labels), 0.001)
        y = tf.fill(tf.shape(labels), pos_weight)
        return tf.where(tf.less(labels, thresh), x, y)

    def get_feed_dict(self, A, L, idx, val, train_dataset, train_labels):
        feed_dict = {self.placeholders['A']: A,
                     self.placeholders['L']: L,
                     self.placeholders['idx']: idx,
                     self.placeholders['val']: val,
                     self.placeholders['train_dataset']: train_dataset,
                     self.placeholders['train_labels']: train_labels}
        return feed_dict

    def get_doc(self, doc_id):
        doc = [np.array(self.corpus['tokenized'][doc_id]).copy()]
        A, L = get_lapl(doc, self.vocab_size).__next__()

        return [sp2tf(A), sp2tf(L)]

    def get_sample(self, ratio=.250):
        """get random sample from corpus graph cache"""
        if (self.samples is None or self.current_sample == len(self.corpus['tokenized'])):
            self.current_sample = 0
            self.samples = np.arange(len(self.corpus['tokenized']))
            np.random.shuffle(self.samples)

        # Get document
        doc = [np.array([])]
        while len(doc[0]) < self.window_size:
            doc = [np.array(self.corpus['tokenized'][self.samples[self.current_sample]]).copy()]
            if self.current_sample == len(self.corpus['tokenized']):
                self.current_samples = 0
                self.samples = np.arange(len(self.corpus['tokenized']))
                np.random.shuffle(self.samples)
            else:
                self.current_sample += 1

        A, L = get_lapl(doc, self.vocab_size).__next__()

        pos_idx = np.random.choice(range(len(A.row)), self.pos_sample_size)

        idx = np.array(list(zip(A.row, A.col)))[pos_idx, :]
        val = A.data[pos_idx]

        # separately generate negative samples
        neg = np.random.randint(self.vocab_size, size=[int(self.pos_sample_size*ratio), 2])

        # concat
        val = np.hstack((val, np.zeros(int(self.pos_sample_size*ratio))))
        idx = np.vstack((idx, neg))

        windows = np.copy(np.lib.stride_tricks.as_strided(doc[0],
                                                          (len(doc[0])-self.window_size,
                                                           self.window_size+1),
                                                          2 * doc[0].strides))
        np.random.shuffle(windows)

        train_dataset = windows[:512, :-1]
        train_labels = windows[:512, -1:]

        return (sp2tf(A), sp2tf(L)), idx, val, train_dataset, train_labels

    def train(self, save_dir, num_epochs=100, print_freq=50, backup_freq=None, friendly_print=False, save_name='model'):
        """train op that can be invoked multiple times."""
        tf.set_random_seed(43)
        np.random.seed(43)

        if (not os.path.isdir(save_dir)):
            raise ValueError("The directory {} does not exists".format(save_dir))

        for e in range(num_epochs):
            self.trained += 1
            (A, L), idx, val, train_dataset, train_labels = self.get_sample()

            feed_dict = self.get_feed_dict(A, L, idx, val, train_dataset, train_labels)

            outs = self.sess.run([self.opt_op, self.loss, self.graph_loss,
                                  self.aux_loss, self.accuracy],
                                 feed_dict=feed_dict)
            avg_loss, graph_loss, aux_loss, avg_acc = outs[1], outs[2], outs[3], outs[4]
            self._loss_vals.append(avg_loss)
            self._acc_vals.append(avg_acc)

            if friendly_print:
                print('\r iter: %d/%d \t graph loss: %.6f \t aux loss: %.3f \t avg_acc: %.3f'
                      % (e+1, num_epochs, graph_loss, aux_loss, np.sum(self._acc_vals[-e:])/e), end='')
                if (e + 1) % print_freq == 0:
                    print('')
            else:
                if (e + 1) % print_freq == 0:
                    print(' iter: %d/%d \t graph loss: %.6f \t aux loss: %.3f \t avg_acc: %.3f'
                          % (e+1, num_epochs, graph_loss, aux_loss, np.sum(self._acc_vals[-e:])/e))

            if backup_freq:
                if (e + 1) % backup_freq == 0:
                    self.save('{1}/{2}_{3}.ckpt'.format(save_dir, save_name, e + 1))

        else:
            print('**** Done training: {} iterations ****'.format(self.trained))
            self.save('{1}/{2}_final.ckpt'.format(save_dir, save_name))

    def forward(self, doc_id):
        A_o, A_i, L_o, L_i = self.get_doc(doc_id)

        feed_dict = self.get_feed_dict(A_o, A_i, L_o, L_i, 0, 0, 0, 0, 0, 0)
        outs = self.sess.run([self.embed_d], feed_dict=feed_dict)

        return outs[0]

    # def get_doc_embedding(self, doc_id, path):
    #     return self.forward(doc_id)
    #     #with open(os.path.join(path, 'db_matrix{0:07}.npy'.format(doc_id)), 'wb') as f:
    #     #    np.save(f, doc_v)

    # def get_doc_embeddings(self, path):
    #     doc_embeddings = np.zeros((len(self.corpus['tokenized']), self.embedding_size_d*self.h_layers[-2]))
    #     for doc_id in range(len(self.corpus['tokenized'])):
    #         if (doc_id+1 % 100) == 0:
    #             print("{} Documents Processed".format(doc_id))
    #         doc_embeddings[doc_id, :] = self.get_doc_embedding(doc_id, path)
    #     with open(os.path.join(path, 'db_matrix.npy'), 'wb') as f:
    #         np.save(f, doc_embeddings)

    def eval_triplets(self, triplets):
        correct = 0
        for i, triplet in enumerate(triplets):
            if (cosine(self.forward(triplet[0]), self.forward(triplet[1])) <
                cosine(self.forward(triplet[0]), self.forward(triplet[2])) and
                cosine(self.forward(triplet[0]), self.forward(triplet[1])) <
                cosine(self.forward(triplet[1]), self.forward(triplet[2]))):
                correct += 1
            if (i + 1) % 1000 == 0:
                print("Accuracy {0:.3f}, Processed {1} triplets".format(correct/(i+1), i+1))

        print("\nAccuracy {0:.3f}".format(correct/len(triplets)))

    def plot(self):
        """Plotting loss function"""
        plt.figure(figsize=(12, 6))
        plt.plot(self._loss_vals, color='red')
        plt.plot(self._acc_vals, color='blue')

        plt.legend(handles=[mpatches.Patch(color='red', label='loss'),
                            mpatches.Patch(color='blue', label='acc')],
                   bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

    def get_reconstruction(self, doc=None):
        if doc:
            # A_o, A_i, L_o, L_i = Doc2Graph(doc, doc_id).doc2graph()
            (A_o, A_i, L_o, L_i), idx_o, idx_i, val_o, val_i = self.get_sample()
        else:
            (A_o, A_i, L_o, L_i), idx_o, idx_i, val_o, val_i = self.get_sample()

        feed_dict = self.get_feed_dict(A_o, A_i, L_o, L_i, idx_o, idx_i, val_o, val_i)
        recon_o, recon_i = self.sess.run([self.recon_o, self.recon_i], feed_dict=feed_dict)
        return A_o, A_i, recon_o, recon_i

    def get_embeddings(self, doc=None, doc_id=None):
        if doc:
            # A_o, A_i, L_o, L_i = 1#Doc2Graph(doc, doc_id).doc2graph()
            (A_o, A_i, L_o, L_i), idx_o, idx_i, val_o, val_i = self.get_sample()
        else:
            (A_o, A_i, L_o, L_i), idx_o, idx_i, val_o, val_i = self.get_sample()

        feed_dict = self.get_feed_dict(A_o, A_i, L_o, L_i, idx_o, idx_i, val_o, val_i)

        emb_o, emb_i = self.sess.run([self.emb_o, self.emb_i], feed_dict=feed_dict)
        return A_o, A_i, emb_o, emb_i

    def save(self, file_name):
        print('Saving model: ', file_name)
        self.saver.save(self.sess, file_name)
        with open(file_name[:-5]+'acc', 'wb') as f:
            np.save(f, np.array(self._acc_vals))
        with open(file_name[:-5]+'loss', 'wb') as f:
            np.save(f, np.array(self._loss_vals))


    def load(self, file_name):
        print('Loading model: ', file_name)
        self.saver.restore(self.sess, file_name)
