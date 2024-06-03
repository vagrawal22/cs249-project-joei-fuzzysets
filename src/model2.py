'''Model for holding TF parts. etc.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
# import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


from multiG import multiG
import pickle
from utils import circular_correlation, np_ccorr

# Orthogonal Initializer from
# https://github.com/OlavHN/bnlstm
def orthogonal(shape):
  flat_shape = (shape[0], np.prod(shape[1:]))
  a = np.random.normal(0.0, 1.0, flat_shape)
  u, _, v = np.linalg.svd(a, full_matrices=False)
  q = u if u.shape == flat_shape else v
  return q.reshape(shape)

def orthogonal_initializer(scale=1.0, dtype=tf.float32):
  def _initializer(shape, dtype=tf.float32, partition_info=None):
    return tf.constant(orthogonal(shape) * scale, dtype)
  return _initializer

class TFParts(object):
    '''TensorFlow-related things. 
    
    This is to keep TensorFlow-related components in a neat shell.
    '''

    def __init__(self, num_rels1, num_ents1, num_rels2, num_ents2, method='fuzzyset', bridge='CG', dim1=300, dim2=100, batch_sizeK1=512, batch_sizeK2=512, batch_sizeA=256, L1=False):
        self._num_relsA = num_rels1
        self._num_entsA = num_ents1
        self._num_relsB = num_rels2
        self._num_entsB = num_ents2
        self.method = method
        self.bridge = bridge
        self._dim1 = dim1
        self._dim2 = dim2
        self._hidden_dim = hid_dim = 50
        self._batch_sizeK1 = batch_sizeK1
        self._batch_sizeK2 = batch_sizeK2
        self._batch_sizeA = batch_sizeA
        self._epoch_loss = 0
        # margins
        self._m1 = 0.5
        self._m2 = 1.0
        self._mA = 0.5
        self.L1 = L1
        self._lr = 0.0005
        self.build()
        print("TFparts build up! Embedding method: [" + self.method + "]. Bridge method:[" + self.bridge + "]")
        print("Margin Paramter: [m1] " + str(self._m1) + " [m2] " + str(self._m2))

    @property
    def dim(self):
        return self._dim1, self._dim2

    def build(self):
        tf.reset_default_graph()

        with tf.variable_scope("graph"):
            # Variables (matrix of embeddings/transformations)
            # KG1
            self._ht1 = ht1 = tf.get_variable(
                name='ht1',  # for t AND h
                shape=[self._num_entsA, self._dim1],
                dtype=tf.float32)
            self._r1 = r1 = tf.get_variable(
                name='r1',
                shape=[self._num_relsA, self._dim1],
                dtype=tf.float32)
            # KG2
            self._ht2 = ht2 = tf.get_variable(
                name='ht2',  # for t AND h
                shape=[self._num_entsB, self._dim2],
                dtype=tf.float32)
            self._r2 = r2 = tf.get_variable(
                name='r2',
                shape=[self._num_relsB, self._dim2],
                dtype=tf.float32)

            tf.summary.histogram("ht1", ht1)
            tf.summary.histogram("ht2", ht2)
            tf.summary.histogram("r1", r1)
            tf.summary.histogram("r2", r2)

            self._ht1_norm = tf.nn.l2_normalize(ht1, 1)
            self._ht2_norm = tf.nn.l2_normalize(ht2, 1)

            ######################## Graph A Loss #######################
            self._A_h_index = A_h_index = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name='A_h_index')
            self._A_r_index = A_r_index = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name='A_r_index')
            self._A_t_index = A_t_index = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name='A_t_index')
            self._A_hn_index = A_hn_index = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name='A_hn_index')
            self._A_tn_index = A_tn_index = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name='A_tn_index')

            A_h_ent_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht1, A_h_index), 1)
            A_t_ent_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht1, A_t_index), 1)
            A_rel_batch = tf.nn.embedding_lookup(r1, A_r_index)

            A_hn_ent_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht1, A_hn_index), 1)
            A_tn_ent_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht1, A_tn_index), 1)

            if self.method == 'fuzzyset':
                ##### Fuzzy Set Embedding score
                def fuzzy_membership(entity, relation, target):
                    return tf.reduce_sum(entity * relation * target, axis=1)

                A_loss_matrix = fuzzy_membership(A_h_ent_batch, A_rel_batch, A_t_ent_batch)
                A_neg_matrix = fuzzy_membership(A_hn_ent_batch, A_rel_batch, A_tn_ent_batch)

                self._A_loss = A_loss = tf.reduce_sum(tf.maximum(tf.add(tf.subtract(A_neg_matrix, A_loss_matrix), self._m1), 0.)) / self._batch_sizeK1

            elif self.method == 'distmult':
                ##### DistMult score
                A_loss_matrix = tf.reduce_sum(tf.multiply(A_rel_batch, tf.multiply(A_h_ent_batch, A_t_ent_batch)), 1)
                A_neg_matrix = tf.reduce_sum(tf.multiply(A_rel_batch, tf.multiply(A_hn_ent_batch, A_tn_ent_batch)), 1)

                self._A_loss = A_loss = tf.reduce_sum(tf.maximum(tf.add(tf.subtract(A_neg_matrix, A_loss_matrix), self._m1), 0.)) / self._batch_sizeK1

            elif self.method == 'hole':
                ##### HolE score
                A_loss_matrix = tf.reduce_sum(tf.multiply(A_rel_batch, circular_correlation(A_h_ent_batch, A_t_ent_batch)), 1)
                A_neg_matrix = tf.reduce_sum(tf.multiply(A_rel_batch, circular_correlation(A_hn_ent_batch, A_tn_ent_batch)), 1)

                self._A_loss = A_loss = tf.reduce_sum(tf.maximum(tf.add(tf.subtract(A_neg_matrix, A_loss_matrix), self._m1), 0.)) / self._batch_sizeK1

            else:
                raise ValueError('Embedding method not valid!')


            ######################## Graph B Loss #######################
            self._B_h_index = B_h_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK2],
                name='B_h_index')
            self._B_r_index = B_r_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK2],
                name='B_r_index')
            self._B_t_index = B_t_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK2],
                name='B_t_index')
            self._B_hn_index = B_hn_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK2],
                name='B_hn_index')
            self._B_tn_index = B_tn_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK2],
                name='B_tn_index')

            B_h_ent_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht2, B_h_index), 1)
            B_t_ent_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht2, B_t_index), 1)
            B_rel_batch = tf.nn.embedding_lookup(r2, B_r_index)

            B_hn_ent_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht2, B_hn_index), 1)
            B_tn_ent_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht2, B_tn_index), 1)

            if self.method == 'fuzzyset':
                #### Fuzzy Set Embedding score
                B_loss_matrix = fuzzy_membership(B_h_ent_batch, B_rel_batch, B_t_ent_batch)
                B_neg_matrix = fuzzy_membership(B_hn_ent_batch, B_rel_batch, B_tn_ent_batch)

                self._B_loss = B_loss = tf.reduce_sum(tf.maximum(tf.add(tf.subtract(B_neg_matrix, B_loss_matrix), self._m2), 0.)) / self._batch_sizeK2

            elif self.method == 'distmult':
                ##### DistMult score
                B_loss_matrix = tf.reduce_sum(tf.multiply(B_rel_batch, tf.multiply(B_h_ent_batch, B_t_ent_batch)), 1)
                B_neg_matrix = tf.reduce_sum(tf.multiply(B_rel_batch, tf.multiply(B_hn_ent_batch, B_tn_ent_batch)), 1)

                self._B_loss = B_loss = tf.reduce_sum(tf.maximum(tf.add(tf.subtract(B_neg_matrix, B_loss_matrix), self._m2), 0.)) / self._batch_sizeK2

            elif self.method == 'hole':
                ##### HolE score
                B_loss_matrix = tf.reduce_sum(tf.multiply(B_rel_batch, circular_correlation(B_h_ent_batch, B_t_ent_batch)), 1)
                B_neg_matrix = tf.reduce_sum(tf.multiply(B_rel_batch, circular_correlation(B_hn_ent_batch, B_tn_ent_batch)), 1)

                self._B_loss = B_loss = tf.reduce_sum(tf.maximum(tf.add(tf.subtract(B_neg_matrix, B_loss_matrix), self._m2), 0.)) / self._batch_sizeK2

            else:
                raise ValueError('Embedding method not valid!')

            tf.summary.scalar("A_loss", A_loss)
            tf.summary.scalar("B_loss", B_loss)
            # ######################## Associative Loss #######################
            # optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)  # Define optimizer here
            # self._train_op_AM = optimizer.minimize(self._A_loss)

            ######################## Combined Loss #######################
            self._J = J = A_loss + B_loss
            tf.summary.scalar("J_loss", J)

            self.summary = tf.summary.merge_all()
            self._lr = lr = tf.placeholder(tf.float32)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            
            # Defining the training operations for both Graph A and Graph B
            self._train_op_A = optimizer.minimize(A_loss)
            self._train_op_B = optimizer.minimize(B_loss)
            
            self._sess = tf.Session()
            self._sess.run(tf.global_variables_initializer())

            self._saver = tf.train.Saver()

    def calc(self, A_h_index, A_r_index, A_t_index, A_hn_index, A_tn_index, B_h_index, B_r_index, B_t_index, B_hn_index, B_tn_index, lr, writemodel=False, modelindex=0):
        '''Calculates loss and applies gradients'''
        loss, _, summary = self._sess.run(
            [self._J, self._op, self.summary],
            feed_dict={
                self._A_h_index: A_h_index,
                self._A_r_index: A_r_index,
                self._A_t_index: A_t_index,
                self._A_hn_index: A_hn_index,
                self._A_tn_index: A_tn_index,
                self._B_h_index: B_h_index,
                self._B_r_index: B_r_index,
                self._B_t_index: B_t_index,
                self._B_hn_index: B_hn_index,
                self._B_tn_index: B_tn_index,
                self._lr: lr
            })
        self._epoch_loss += loss
        return loss, summary

    def normalize(self):
        '''Normalizes the entity embeddings'''
        self._sess.run(
            [self._ht1.assign(self._ht1_norm), self._ht2.assign(self._ht2_norm)])

    def getJ(self):
        return self._epoch_loss

    def resetJ(self):
        self._epoch_loss = 0

    def save_embeddings(self, sess, save_path):
        variables_to_save = {
            'ht1': self._ht1,
            'r1': self._r1,
            'ht2': self._ht2,
            'r2': self._r2
        }
        saver = tf.train.Saver(variables_to_save)
        saver.save(sess, save_path)


