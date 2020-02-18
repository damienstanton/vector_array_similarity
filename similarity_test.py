import numpy as np
import unittest
from similarity import SimMetrics, vector_array_similarity


def similarity_for_test(sim_metrics):
    return vector_array_similarity(sim_metrics)


def double_weighted_using_min_obs_similarity_with_mean(sim_metrics):
    met = sim_metrics.normalize_obs()
    weights = np.array([[min(a, b) for b in met.n_obs_2] for a in met.n_obs_1])
    weighted_sims = met.similarity_matrix * weights
    one_way = weighted_sims.max(axis=1).sum()
    other_way = weighted_sims.max(axis=0).sum()
    return (one_way + other_way) / 2


def unweighted_similarity(sim_metrics):
    one_way = np.mean(sim_metrics.similarity_matrix.max(axis=0))
    other_way = np.mean(sim_metrics.similarity_matrix.max(axis=1))
    return (one_way + other_way) / 2


def unweighted_harmonic_similarity(sim_metrics):
    one_way = np.mean(sim_metrics.similarity_matrix.max(axis=0))
    other_way = np.mean(sim_metrics.similarity_matrix.max(axis=1))
    if one_way == 0.0 or other_way == 0.0:
        return 0.0
    return 2 / (1 / one_way + 1 / other_way)


def weighted_similarity(sim_metrics):
    sim_metrics = sim_metrics.normalize_obs()
    weighted_one_way = (
        sim_metrics.similarity_matrix.max(axis=1) * sim_metrics.n_obs_1
    ).sum()
    weighted_other_way = (
        sim_metrics.similarity_matrix.max(axis=0) * sim_metrics.n_obs_2
    ).sum()
    return (weighted_one_way + weighted_other_way) / 2


def harmonic_weighted_similarity(sim_metrics):
    sim_metrics = sim_metrics.normalize_obs()
    weighted_one_way = (
        sim_metrics.similarity_matrix.max(axis=1) * sim_metrics.n_obs_1
    ).sum()
    weighted_other_way = (
        sim_metrics.similarity_matrix.max(axis=0) * sim_metrics.n_obs_2
    ).sum()
    if weighted_one_way == 0.0 or weighted_other_way == 0.0:
        return 0.0
    return 2 / (1 / weighted_one_way + 1 / weighted_other_way)


class SimilarityPropertiesTest(unittest.TestCase):
    def test_self_similarity_is_one(self):
        # similarity_matrix if finding (A, B) ~ (A, B) and A ~ B == 0.0
        discrete_clusters = SimMetrics([[1.0, 0.0], [0.0, 1.0]], [7, 3], [7, 3])
        # similarity_matrix if finding (A, B) ~ (A, B) and A ~ B == 0.1
        non_discrete_clusters = SimMetrics([[1.0, 0.1], [0.1, 1.0]], [7, 3], [7, 3])
        self.assertEqual(similarity_for_test(discrete_clusters), 1.0)
        self.assertEqual(similarity_for_test(non_discrete_clusters), 1.0)

    def test_proportional_self_similarity_is_one(self):
        # want a high score comparing one week with three weeks (for example)
        # if comparing a super regular observer/signal to itself
        discrete_clusters = SimMetrics([[1.0, 0.0], [0.0, 1.0]], [7, 3], [21, 9])
        non_discrete_clusters = SimMetrics([[1.0, 0.1], [0.1, 1.0]], [7, 3], [21, 9])
        self.assertEqual(similarity_for_test(discrete_clusters), 1.0)
        self.assertEqual(similarity_for_test(non_discrete_clusters), 1.0)

    def test_similarity_is_symmetric(self):
        # Naive Avg(maxSim) won't satisfy symmetric requirement for similarity.
        # Ex:  Find (A, B) ~ (a, b) if
        # A ~ a = .2
        # A ~ b = .3
        # B ~ a = .1
        # B ~ b = 0
        # Then, from the perspective of (A, B), A's best match is b and B's best match is a,
        # so we'd be taking Avg(.3, .1), but from the perspective of (a, b), a's best match is A,
        # and b's best match is also A, so we'd be taking Avg(.2, .3)
        m = SimMetrics([[0.2, 0.3], [0.1, 0.0]], [7, 3], [10, 2])
        self.assertEqual(similarity_for_test(m), similarity_for_test(m.transpose()))
        m2 = SimMetrics([[0.2, 0.3, 0], [0.1, 0.0, 0]], [7, 3], [10, 2, 10])
        self.assertEqual(similarity_for_test(m2), similarity_for_test(m2.transpose()))

    def test_extra_clusters_decrease_similarity(self):
        # (A, B) ~ (A, B, C) < 1.0
        self.assertTrue(
            similarity_for_test(SimMetrics([[1, 0, 0], [0, 1, 0]], [7, 3], [7, 3, 5]))
            < 1.0
        )
        # (A) ~ (A, B) > (A, C) ~ (A, B)
        A_AB = SimMetrics([[1, 0]], [1], [1, 1])
        AC_AB = SimMetrics([[1, 0], [0, 0]], [1, 1], [1, 1])
        self.assertTrue(similarity_for_test(A_AB) > similarity_for_test(AC_AB))

    def test_similarity_of_extra_cluster_matters(self):
        # (A, B) ~ (A, B, C) should be greater if C is similar to A and/or B
        # challenge:  reconcile this with
        # test_self_similarity_is_one for non_discrete_clusters
        not_similar_C = SimMetrics([[1, 0, 0], [0, 1, 0]], [7, 3], [7, 3, 5])
        similar_C = SimMetrics([[1, 0, 0.3], [0, 1, 0.4]], [7, 3], [7, 3, 5])
        self.assertTrue(
            similarity_for_test(not_similar_C) < similarity_for_test(similar_C)
        )

    def test_trivial_comparison(self):
        # (A) ~ (B) should equal A ~ B
        self.assertEqual(similarity_for_test(SimMetrics([[0.0]], [10], [15])), 0.0)
        self.assertEqual(similarity_for_test(SimMetrics([[0.3]], [2], [2])), 0.3)
        self.assertEqual(similarity_for_test(SimMetrics([[0.7]], [100], [5])), 0.7)
        self.assertEqual(similarity_for_test(SimMetrics([[1.0]], [18], [4])), 1.0)

    def test_noise_canceling_effect_of_weighting(self):
        matrix = [[0.98, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]]
        n_obs_1 = [1000]
        n_obs_2 = [590, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        with_noise = SimMetrics(matrix, n_obs_1, n_obs_2)
        without_noise = SimMetrics([[0.98]], [1000], [590])
        self.assertTrue(
            similarity_for_test(with_noise) < similarity_for_test(without_noise)
        )
        self.assertTrue(
            similarity_for_test(with_noise) / similarity_for_test(without_noise) > 0.9
        )

    def test_against_step_function_near_zero(self):
        actually_zero = SimMetrics([[1, 0], [0, 0]], [1, 1], [1, 1])
        close_to_zero = SimMetrics(
            [[1, 0.000000001], [0.000000001, 0.000000001]], [1, 1], [1, 1]
        )
        self.assertTrue(
            similarity_for_test(actually_zero) < similarity_for_test(close_to_zero)
        )
        self.assertTrue(
            similarity_for_test(actually_zero) / similarity_for_test(close_to_zero)
            > 0.9
        )

    def test_limited_effect_of_small_overlap(self):
        # When most underlying observations from both arrays are in dissimilar clusters
        # having a brief "bump in the night" meeting shouldn't unduly boost their
        # similarity
        metrics = SimMetrics([[0.9, 0.01], [0.01, 0.9]], [999, 1], [1, 999])
        self.assertTrue(similarity_for_test(metrics) < 0.1)

    def test_added_high_similarity_increases_similarity(self):
        # A.n_obs = 500
        # B.n_obs = 999
        # C.n_obs = 1
        # A ~ B = .9
        # A ~ C = 1 (identical)
        # (A) ~ (B, C) should be greater than (A) ~ (B)
        # (not that B and C would likely be in an array together without merging)
        self.assertTrue(
            similarity_for_test(SimMetrics([[0.9]], [500], [999]))
            < similarity_for_test(SimMetrics([[0.9, 1]], [500], [999, 1]))
        )

    def test_order_of_clusters_does_not_matter(self):
        # A ~ a = 0.1
        # A ~ b = 0.2
        # A ~ c = 0.6
        # B ~ a = 0.9
        # B ~ b = 0.8
        # B ~ c = 0.0
        # A.n_obs = 10
        # B.n_obs = 20
        # a.n_obs = 2
        # b.n_obs = 8
        # c.n_obs = 100
        # (A, B) ~ (a, b, c) = (B, A) ~ (b, c, a) and all permutations generally
        AB_abc = SimMetrics([[0.1, 0.2, 0.6], [0.9, 0.8, 0.0]], [10, 20], [2, 8, 100])
        AB_acb = SimMetrics([[0.1, 0.6, 0.2], [0.9, 0.0, 0.8]], [10, 20], [2, 100, 8])
        BA_bca = SimMetrics([[0.8, 0.0, 0.9], [0.2, 0.6, 0.1]], [20, 10], [8, 100, 2])
        BA_cba = SimMetrics([[0.0, 0.8, 0.9], [0.6, 0.2, 0.1]], [20, 10], [100, 8, 2])
        self.assertEqual(similarity_for_test(AB_abc), similarity_for_test(AB_acb))
        self.assertEqual(similarity_for_test(AB_abc), similarity_for_test(BA_bca))
        self.assertEqual(similarity_for_test(AB_abc), similarity_for_test(BA_cba))

    def test_jaccard_interpretation(self):
        # Given equal weights and either 1 or 0 for cluster similarity, can
        # the cluster array similarity metric match jaccard for sets?
        self.assertEqual(similarity_for_test(SimMetrics([[1, 0]], [1], [1, 1])), 0.5)
        self.assertEqual(
            similarity_for_test(SimMetrics([[1, 0, 0]], [1], [1, 1, 1])), 1 / 3
        )
        self.assertEqual(
            similarity_for_test(SimMetrics([[1, 0], [0, 0]], [1, 1], [1, 1])), 1 / 3
        )


if __name__ == "__main__":
    unittest.main()
