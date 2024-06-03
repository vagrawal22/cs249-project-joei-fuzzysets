from __future__ import absolute_import, division, print_function
import numpy as np
import pickle
import time
from KG import KG

from fuzzyKG import FuzzyKG

class multiG(object):
    '''This class stores two KGs and alignment seeds. Initialize KGs separately because sometimes we don't load word embeddings'''

    def __init__(self, KG1=None, KG2=None):
        if KG1 is None or KG2 is None:
            self.KG1 = KG()
            self.KG2 = KG()
        else:
            self.KG1 = KG1
            self.KG2 = KG2
        self.lan1 = 'en'
        self.lan2 = 'fr'
        self.align = np.array([0])
        self.align_desc = np.array([0])
        self.aligned_KG1 = set([])
        self.aligned_KG2 = set([])
        self.aligned_KG1_index = np.array([0])
        self.aligned_KG2_index = np.array([0])
        self.unaligned_KG1_index = np.array([0])
        self.unaligned_KG2_index = np.array([0])
        self.align_valid = np.array([0])
        self.n_align = 0
        self.n_align_desc = 0
        self.ent12 = {}
        self.ent21 = {}
        self.batch_sizeK1 = 1024
        self.batch_sizeK2 = 64
        self.batch_sizeA = 32
        self.L1 = False
        self.dim1 = 300  # stored for TF_Part
        self.dim2 = 100
        self.fuzzy_kg = FuzzyKG()

    def load_fuzzy_sets(self, entity_file, relation_file):
        # Load fuzzy sets for entities and relations using FuzzyKG
        self.fuzzy_kg.load_fuzzy_sets(entity_file, relation_file)

    def calculate_entity_similarity(self, entity1, entity2):
        # Calculate similarity between two entities based on fuzzy sets using FuzzyKG
        return self.fuzzy_kg.calculate_entity_similarity(entity1, entity2)

    def calculate_relation_similarity(self, relation1, relation2):
        # Calculate similarity between two relations based on fuzzy sets using FuzzyKG
        return self.fuzzy_kg.calculate_relation_similarity(relation1, relation2)

    def load_align(self, filename, lan1='en', lan2='fr', splitter='@@@', line_end='\n', desc=False):
        '''Load the dataset.'''
        self.n_align = 0
        self.n_align_desc = 0
        self.align = []
        if desc:
            self.align_desc = []
        for line in open(filename, encoding='utf-8'):
            line = line.rstrip(line_end).split(splitter)
            e1 = self.KG1.ent_str2index(line[0])
            e2 = self.KG2.ent_str2index(line[2])
            if e1 is None or e2 is None:
                continue
            self.align.append((e1, e2))
            self.aligned_KG1.add(e1)
            self.aligned_KG2.add(e2)
            if self.ent12.get(e1) is None:
                self.ent12[e1] = set([e2])
            else:
                self.ent12[e1].add(e2)
            if self.ent21.get(e2) is None:
                self.ent21[e2] = set([e1])
            else:
                self.ent21[e2].add(e1)
            self.n_align += 1
            if desc:
                if self.KG1.get_desc_embed(e1) is not None and self.KG2.get_desc_embed(e2) is not None:
                    self.align_desc.append((e1, e2))
                    self.n_align_desc += 1
        self.align = np.array(self.align)
        if desc:
            self.align_desc = np.array(self.align_desc)
        self.aligned_KG1_index = np.array([e for e in self.aligned_KG1])
        self.aligned_KG2_index = np.array([e for e in self.aligned_KG2])
        self.unaligned_KG1_index = np.array([i for i in self.KG1.desc_index if i not in self.aligned_KG1])
        self.unaligned_KG2_index = np.array([i for i in self.KG2.desc_index if i not in self.aligned_KG2])
        print("Loaded aligned entities from", filename, ". #pairs:", self.n_align)

    def load_valid(self, filename, size=1024, lan1='en', lan2='fr', splitter='@@@', line_end='\n', desc=False):
        '''Load the dataset.'''
        self.align_valid = []
        for line in open(filename, encoding='utf-8'):
            line = line.rstrip(line_end).split(splitter)
            e1 = self.KG1.ent_str2index(line[0])
            e2 = self.KG2.ent_str2index(line[1])
            if e1 is None or e2 is None:
                continue
            if self.ent12.get(e1) is None:
                self.ent12[e1] = set([e2])
            else:
                self.ent12[e1].add(e2)
            if self.ent21.get(e2) is None:
                self.ent21[e2] = set([e1])
            else:
                self.ent21[e2].add(e1)
            if self.KG1.get_desc_embed(e1) is not None and self.KG2.get_desc_embed(e2) is not None:
                self.align_valid.append((e1, e2))
                if len(self.align_valid) >= size:
                    break
        self.align_valid = np.array(self.align_valid)
        print("Loaded validation entities from", filename, ". #pairs:", size)

    def load_more_gt(self, filename):
        for line in open(filename):
            line = line.rstrip().split(splitter)
            e1 = self.KG1.ent_str2index(line[0])
            e2 = self.KG2.ent_str2index(line[1])
            if e1 is None or e2 is None:
                continue
            if self.ent12.get(e1) is None:
                self.ent12[e1] = set([e2])
            else:
                self.ent12[e1].add(e2)
            if self.ent21.get(e2) is None:
                self.ent21[e2] = set([e1])
            else:
                self.ent21[e2].add(e1)
        print("Loaded more gt file for negative sampling from", filename)

    def num_align(self):
        '''Returns number of aligned entities.'''
        return self.n_align
    
    def num_align_desc(self):
        '''Returns number of aligned entities with descriptions.'''
        return self.n_align_desc

    def corrupt_desc_pos(self, align, pos, sample_global=True):
        assert pos in [0, 1]
        hit = True
        while hit:
            res = np.copy(align)
            if pos == 0:
                samp = np.random.choice(self.KG1.desc_index) if sample_global else np.random.choice(self.aligned_KG1_index)
                if samp not in self.ent21[align[1]]:
                    hit = False
                    res = np.array([samp, align[1]])
            else:
                samp = np.random.choice(self.KG2.desc_index) if sample_global else np.random.choice(self.aligned_KG2_index)
                if samp not in self.ent12[align[0]]:
                    hit = False
                    res = np.array([align[0], samp])
        return res

    def corrupt_desc(self, align, tar=None):
        pos = tar if tar is not None else np.random.randint(2)
        return self.corrupt_desc_pos(align, pos)
    
    def corrupt_align_pos(self, align, pos):
        assert pos in [0, 1]
        hit = True
        while hit:
            res = np.copy(align)
            if pos == 0:
                samp = np.random.randint(self.KG1.num_ents())
                if samp not in self.ent21[align[1]]:
                    hit = False
                    res = np.array([samp, align[1]])
            else:
                samp = np.random.randint(self.KG2.num_ents())
                if samp not in self.ent12[align[0]]:
                    hit = False
                    res = np.array([align[0], samp])
        return res

    def corrupt_align(self, align, tar=None):
        pos = tar if tar is not None else np.random.randint(2)
        return self.corrupt_align_pos(align, pos)
    
    def corrupt_desc_batch(self, a_batch, tar=None):
        np.random.seed(int(time.time()))
        return np.array([self.corrupt_desc(a, tar) for a in a_batch])

    def corrupt_align_batch(self, a_batch, tar=None):
        np.random.seed(int(time.time()))
        return np.array([self.corrupt_align(a, tar) for a in a_batch])
    
    def sample_false_pair(self, batch_sizeA):
        a = np.random.choice(self.unaligned_KG1_index, batch_sizeA)
        b = np.random.choice(self.unaligned_KG2_index, batch_sizeA)
        return np.array([(a[i], b[i]) for i in range(batch_sizeA)])
    
    def expand_align(self, list_of_pairs):
        # TODO: Implement the expansion logic if needed
        pass
    
    def token_overlap(self, set1, set2):
        min_len = min(len(set1), len(set2))
        hit = sum(1 for tk in set1 if tk in set2)
        return hit / min_len

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
        print("Save data object as", filename)

    def load(self, filename):
        with open(filename, 'rb') as f:
            tmp_dict = pickle.load(f)
            self.__dict__.update(tmp_dict)
        print("Loaded data object from", filename)
