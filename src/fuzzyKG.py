import numpy as np

class FuzzyKG(object):
    def __init__(self):
        self.entities = {}  # Dictionary to store fuzzy sets for entities
        self.relations = {}  # Dictionary to store fuzzy sets for relations
    
    def load_fuzzy_sets(self, entity_file, relation_file):
        # Load fuzzy sets for entities
        with open(entity_file, 'r') as f:
            for line in f:
                entity, fuzzy_set = line.strip().split('\t')
                self.entities[entity] = self._parse_fuzzy_set(fuzzy_set)
        
        # Load fuzzy sets for relations
        with open(relation_file, 'r') as f:
            for line in f:
                relation, fuzzy_set = line.strip().split('\t')
                self.relations[relation] = self._parse_fuzzy_set(fuzzy_set)
    
    def _parse_fuzzy_set(self, fuzzy_set_str):
        # Convert string representation of fuzzy set to numpy array
        # This is a simple example, you might have a more complex parsing logic
        return np.array([float(x) for x in fuzzy_set_str.split(',')])
    
    def calculate_entity_similarity(self, entity1, entity2):
        # Calculate similarity between two entities based on their fuzzy sets
        set1 = self.entities.get(entity1, np.zeros_like(next(iter(self.entities.values()))))
        set2 = self.entities.get(entity2, np.zeros_like(next(iter(self.entities.values()))))
        # Example similarity measure: cosine similarity
        similarity = np.dot(set1, set2) / (np.linalg.norm(set1) * np.linalg.norm(set2))
        return similarity
    
    def calculate_relation_similarity(self, relation1, relation2):
        # Calculate similarity between two relations based on their fuzzy sets
        set1 = self.relations.get(relation1, np.zeros_like(next(iter(self.relations.values()))))
        set2 = self.relations.get(relation2, np.zeros_like(next(iter(self.relations.values()))))
        # Example similarity measure: cosine similarity
        similarity = np.dot(set1, set2) / (np.linalg.norm(set1) * np.linalg.norm(set2))
        return similarity
