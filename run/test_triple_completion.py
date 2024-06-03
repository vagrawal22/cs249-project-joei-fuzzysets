from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), './src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from KG import KG 

from utils import circular_correlation, np_ccorr  # Make sure you have these functions available

class TFParts(object):
    '''TensorFlow-related things. 
    
    This is to keep TensorFlow-related components in a neat shell.
    '''
    def __init__(self, num_rels1, num_ents1, num_rels2, num_ents2, method='fuzzyset', bridge='CMP-linear', dim1=300, dim2=100, batch_sizeK1=512, batch_sizeK2=512, batch_sizeA=256, L1=False):
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

            # Optimizer
            self._lr = lr = tf.placeholder(tf.float32)
            self._opt = opt = tf.train.AdamOptimizer(lr)
            self._train_op_A = train_op_A = opt.minimize(self._A_loss)
            self._train_op_B = train_op_B = opt.minimize(self._B_loss)
            self._train_op_AM = train_op_AM = opt.minimize(self._AM_loss)

            # Saver
            self.summary_op = tf.summary.merge_all()
            self._saver = tf.train.Saver()

def load_model(session, model_path):
    saver = tf.train.import_meta_graph(model_path + '.meta')
    saver.restore(session, model_path)
    graph = tf.get_default_graph()
    return graph

def get_embedding_tensors(graph):
    ht1 = graph.get_tensor_by_name('graph/ht1:0')
    r1 = graph.get_tensor_by_name('graph/r1:0')
    ht2 = graph.get_tensor_by_name('graph/ht2:0')
    r2 = graph.get_tensor_by_name('graph/r2:0')
    return ht1, r1, ht2, r2

def calculate_scores_batch(ht, r, h_indices, r_indices, method='transe'):
    h_embeddings = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht, h_indices), axis=1)
    r_embeddings = tf.nn.l2_normalize(tf.nn.embedding_lookup(r, r_indices), axis=1)
    t_embeddings = ht

    if method == 'transe':
        scores = -tf.norm(tf.expand_dims(h_embeddings + r_embeddings, 1) - t_embeddings, axis=2)
    elif method == 'distmult':
        scores = tf.reduce_sum(tf.expand_dims(h_embeddings * r_embeddings, 1) * t_embeddings, axis=2)
    elif method == 'hole':
        scores = tf.reduce_sum(tf.expand_dims(r_embeddings, 1) * circular_correlation(h_embeddings, t_embeddings), axis=2)
    elif method == 'fuzzyset': # modify this line
        def fuzzy_membership(entity, relation, target):
            membership_degree = tf.minimum(tf.minimum(entity, relation), target)
            return tf.reduce_sum(membership_degree, axis=1)
        scores = fuzzy_membership(h_embeddings, r_embeddings, t_embeddings)
        # scores = tf.reduce_sum(h_embedding * r_embedding * t_embeddings, axis=1)
    else:
        raise ValueError('Embedding method not valid!')

    return scores

def calculate_mrr_hits(session, graph, test_triples, ht, r, batch_size=100, method='transe'):
    # filtered_triples = [triple for triple in test_triples if 0 <= triple[1] < 34]
    # filtered_triples = [triple for triple in filtered_triples if 0 <= triple[0] < 26039]
    mrr = 0.0
    hits_at_5 = 0
    hits_at_10 = 0
    num_triples = len(test_triples)
    count = 0 
    print("shape of heads",ht.shape)
    print("shape of r",r.shape)

    num_batches = (num_triples + batch_size - 1) // batch_size  # Ceiling division

    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, num_triples)
        batch_triples = test_triples[batch_start:batch_end]

        h_indices, r_indices, t_indices = zip(*batch_triples)
        h_indices = np.array(h_indices)
        r_indices = np.array(r_indices)
        t_indices = np.array(t_indices)

        scores = session.run(calculate_scores_batch(ht, r, h_indices, r_indices, method))

        for j in range(batch_end - batch_start):
            t_index = t_indices[j]
            ranked_indices = np.argsort(-scores[j])  # Sort in descending order
            # # rank = np.where(ranked_indices == t_index)[0][0] + 1  # 1-based rank

            # mrr += 1.0 / rank
            # if rank <= 5:
            #     hits_at_5 += 1
            # if rank <= 10:
            #     hits_at_10 += 1

            if t_index in ranked_indices:
                rank = np.where(ranked_indices == t_index)[0][0] + 1  # 1-based rank

                mrr += 1.0 / rank
                if rank <= 5:
                    hits_at_5 += 1
                if rank <= 10:
                    hits_at_10 += 1
                count += 1
                triples = count+batch_size 
                if (triples) % 100 == 0:
                    print("rank", rank)
                    print(triples)
                    print("mrr", mrr)
            else:
                print(f"Warning: t_index {t_index} not found in ranked_indices for batch {i}, index {j}")

        if (i + 1) % 10 == 0:
            print(f'Processed {batch_end} triples out of {num_triples}')

    mrr /= num_triples
    hits_at_5 /= num_triples
    hits_at_10 /= num_triples

    return mrr, hits_at_5, hits_at_10


model_path = './fuzzy_embeddings.ckpt'

KG1 = KG()

KG1.load_triples(filename = "./data/dbpedia/db_insnet_test.txt", splitter = '\t', line_end = '\n')

test_triples = KG1.triples
kg = 'inst'  # Specify which KG ('A' or 'B')

with tf.Session() as session:
    graph = load_model(session, model_path)
    ht1, r1, ht2, r2 = get_embedding_tensors(graph)
        
    if kg == 'inst':
        ht, r = ht1, r1
    elif kg == 'onto':
        ht, r = ht2, r2
    else:
        raise ValueError('Invalid KG specified. Use "A" or "B".')

    mrr, hits_at_5, hits_at_10 = calculate_mrr_hits(session, graph, test_triples, ht, r, method='transe')
    print("MRR:", mrr)
    print("Hits@5:", hits_at_5)
    print("Hits@10:", hits_at_10)