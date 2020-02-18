import numpy as np

def vector_array_similarity(sim_metrics, eps=0.1):
    score = double_weighted_using_min_weights_similarity_with_max(sim_metrics)
    decay = zeros_decay(sim_metrics.similarity_matrix, eps)
    return score * decay

def double_weighted_using_min_weights_similarity_with_max(sim_metrics):
    met = sim_metrics.normalize_weights()
    weights = np.array([[min(a, b) for b in met.weights_2] for a in met.weights_1])
    weighted_sims = met.similarity_matrix * weights
    one_way = weighted_sims.max(axis=1).sum()
    other_way = weighted_sims.max(axis=0).sum()
    return max(one_way, other_way)

def zeros_decay(mm, eps):
    one_way = zeros_decay_one_way(mm.sum(axis=1), eps)
    other_way = zeros_decay_one_way(mm.sum(axis=0), eps)
    return max(one_way, other_way)

def zeros_decay_one_way(a, eps):
    return ((a > eps).sum() + 1) / (len(a) + 1)

def cosine_similarity(d1, d2):
    dot = sum([d1[k] * d2[k] for k in d1 if k in d2])
    if dot == 0:
        return 0
    mag1 = sum([v**2 for v in d1.values()])**.5
    mag2 = sum([v**2 for v in d2.values()])**.5
    return dot / (mag1 * mag2)

# Contains the similarity matrix between two vector arrays.
# Weights can be assigned to each vector.
# Arrays need not be of the same lenght
class WeightedSimilarityMatrix:
    def __init__(self, similarity_matrix, weights_1=None, weights_2=None):
        # The order of weights_1 should match the row order,
        # and the order of weights_2 should match the column order.
        self.similarity_matrix = np.array(similarity_matrix)
        self.weights_1 = None if weights_1 is None else np.array(weights_1)
        self.weights_2 = None if weights_2 is None else np.array(weights_2)

    def transpose(self):
        return type(self)(self.similarity_matrix.T, self.weights_2, self.weights_1)

    def normalize_weights(self):
        weights_1 = self.weights_1 / self.weights_1.sum()
        weights_2 = self.weights_2 / self.weights_2.sum()
        return type(self)(self.similarity_matrix, weights_1, weights_2)
