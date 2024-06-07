from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), './src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
#sys.path.append('./src')
import os
if not os.path.exists('./results'):
    os.makedirs('./results')

if not os.path.exists('./results/detail'):
    os.makedirs('./results/detail')

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from numpy import linalg as LA
import tensorflow as tf
import time
import multiG  
import model2 as model
from tester1_fuzzy import Tester
import argparse

# all parameter required
parser = argparse.ArgumentParser(description='JOIE Testing: Type Linking')
parser.add_argument('--modelname', type=str,help='model category')
parser.add_argument('--model', type=str,help='model name including data and model')
parser.add_argument('--testfile', type=str,help='test data')
parser.add_argument('--method', type=str,help='embedding method used')
args = parser.parse_args()

path_prefix = './model/'+args.modelname
hparams_str = args.model
model_file = path_prefix+"/"+hparams_str+"/"+args.method+'-model-m2.ckpt'
data_file = path_prefix+"/"+hparams_str+"/"+args.method+'-multiG-m2.bin'
test_data = args.testfile
limit_align_file = None
result_file = path_prefix+"/"+hparams_str+'/detail_result_m2.txt'

topK = 10
max_check = 100000

#dup_set = set([])
#for line in open(old_data):
#    dup_set.add(line.rstrip().split('@@@')[0])

tester = Tester()
tester.build(save_path=model_file, data_save_path=data_file)
# tester
tester.load_test_type(test_data, splitter='\t', line_end='\n')
#tester.load_except_data(except_data, splitter = '@@@', line_end = '\n')

test_id_limits = None
if limit_align_file is not None:
    _, test_id_limits = tester.load_align_ids(limit_align_file, splitter='\t', line_end='\n')

t0 = time.time()

rst_predict = [] # scores for each case
rank_record = []
prop_record = []
print("test_align",tester.test_align)

for id, (e1, e2) in enumerate(tester.test_align):
    if id > 0 and id % 200 == 0:
        print("Tested %d in %d seconds." % (id + 1, time.time() - t0))
        try:
            print(np.mean(rst_predict, axis=0))
        except:
            pass
    vec_proj_e1 = tester.projection(e1, source=1)
    vec_pool_e2 = tester.vec_e[2]
    rst = tester.kNN(vec_proj_e1, vec_pool_e2, topK, limit_ids=test_id_limits)
    this_hit = []
    hit = 0.
    strl = tester.ent_index2str(rst[0][0], 2)
    strr = tester.ent_index2str(e2, 2)
    this_index = 0
    this_rank = None
    for pr in rst:
        this_index += 1
        if (hit < 1. and (pr[0] == e2 or pr[0] in tester.lr_map[e1])) or (hit < 1. and tester.ent_index2str(pr[0], 2) == strr):
            hit = 1.
            this_rank = this_index
        this_hit.append(hit)
    hit_first = 0
    if rst[0][0] == e2 or rst[0][0] in tester.lr_map[e1] or strl == strr:
        hit_first = 1
    if this_rank is None:
        this_rank = tester.rank_index_from(vec_proj_e1, vec_pool_e2, e2, limit_ids=test_id_limits)
    if this_rank > max_check:
        continue
    rst_predict.append(np.array(this_hit))
    rank_record.append(1.0 / (1.0 * this_rank))
    prop_record.append((hit_first, rst[0][1], strl, strr))

mean_rank = np.mean(rank_record)
hits = np.mean(rst_predict, axis=0)

# print out result file
fp = open(result_file, 'w')

fp.write("Mean Rank\n")
print("Mean Rank", mean_rank)
print("Hits@",str(topK))
print("hits",hits)
fp.write(str(mean_rank)+'\n')
fp.write("Hits@"+str(topK)+'\n')
# fp.write(' '.join([str(x) for x in hits]) + '\n')
fp.close()
