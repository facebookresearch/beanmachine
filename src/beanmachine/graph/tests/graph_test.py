# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import numpy as np
from beanmachine import graph


class TestBayesNet(unittest.TestCase):
    def test_simple_dep(self):
        g = graph.Graph()
        c1 = g.add_constant_col_simplex_matrix(np.array([0.8, 0.2]))
        d1 = g.add_distribution(
            graph.DistributionType.TABULAR, graph.AtomicType.BOOLEAN, [c1]
        )
        g.add_operator(graph.OperatorType.SAMPLE, [d1])

    def test_tabular(self):
        g = graph.Graph()
        c1 = g.add_constant_col_simplex_matrix(np.array([0.8, 0.2]))

        # negative test
        with self.assertRaises(ValueError) as cm:
            g.add_distribution(
                graph.DistributionType.TABULAR, graph.AtomicType.BOOLEAN, []
            )
        self.assertTrue("must be COL_SIMPLEX" in str(cm.exception))

        g = graph.Graph()
        c1 = g.add_constant_col_simplex_matrix(np.array([0.8, 0.2]))
        var1 = g.add_operator(
            graph.OperatorType.SAMPLE,
            [
                g.add_distribution(
                    graph.DistributionType.TABULAR, graph.AtomicType.BOOLEAN, [c1]
                )
            ],
        )
        var2 = g.add_operator(
            graph.OperatorType.SAMPLE,
            [
                g.add_distribution(
                    graph.DistributionType.TABULAR, graph.AtomicType.BOOLEAN, [c1]
                )
            ],
        )

        # since the following has two parents it must have a tabular dist with
        # 3 dimensions in the tensor
        with self.assertRaises(ValueError) as cm:
            g.add_operator(
                graph.OperatorType.SAMPLE,
                [
                    g.add_distribution(
                        graph.DistributionType.TABULAR,
                        graph.AtomicType.BOOLEAN,
                        [c1, var1, var2],
                    )
                ],
            )
        self.assertTrue("expected 4 dims got 1" in str(cm.exception))

        c2 = g.add_constant_col_simplex_matrix(np.array([[0.6, 0.99], [0.4, 0.01]]))
        g.add_distribution(
            graph.DistributionType.TABULAR,
            graph.AtomicType.BOOLEAN,
            [c2, g.add_constant(True)],
        )

        with self.assertRaises(ValueError) as cm:
            g.add_distribution(
                graph.DistributionType.TABULAR,
                graph.AtomicType.BOOLEAN,
                [c2, g.add_constant(1)],
            )
        self.assertTrue("only supports boolean parents" in str(cm.exception))

        c3 = g.add_constant_real_matrix(np.array([1.1, -0.1]))
        with self.assertRaises(ValueError) as cm:
            g.add_distribution(
                graph.DistributionType.TABULAR, graph.AtomicType.BOOLEAN, [c3]
            )
        self.assertTrue("must be COL_SIMPLEX" in str(cm.exception))

        c4 = g.add_constant_col_simplex_matrix(np.array([0.6, 0.3, 0.1]))
        with self.assertRaises(ValueError) as cm:
            g.add_distribution(
                graph.DistributionType.TABULAR, graph.AtomicType.BOOLEAN, [c4]
            )
        self.assertTrue("must have two rows" in str(cm.exception))

    def test_bernoulli(self):
        g = graph.Graph()
        c1 = g.add_constant_probability(1.0)
        c2 = g.add_constant_probability(0.8)

        # negative tests on number of parents
        # 0 parents not allowed
        with self.assertRaises(ValueError) as cm:
            g.add_distribution(
                graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, []
            )
        self.assertTrue(
            "Bernoulli distribution must have exactly one parent" in str(cm.exception)
        )
        # 2 parents not allowed
        with self.assertRaises(ValueError) as cm:
            g.add_distribution(
                graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, [c1, c2]
            )
        self.assertTrue(
            "Bernoulli distribution must have exactly one parent" in str(cm.exception)
        )

        # 1 parent is OK
        d1 = g.add_distribution(
            graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, [c1]
        )

        # negative test on type of parent
        c3 = g.add_constant(1)
        with self.assertRaises(ValueError) as cm:
            g.add_distribution(
                graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, [c3]
            )
        self.assertTrue("must be a probability" in str(cm.exception))

        # negative test on value of parent
        with self.assertRaises(ValueError) as cm:
            g.add_constant_probability(1.1)
        self.assertTrue("must be between 0 and 1" in str(cm.exception))

        v1 = g.add_operator(graph.OperatorType.SAMPLE, [d1])
        g.query(v1)
        samples = g.infer(1)
        self.assertEqual(type(samples[0][0]), bool)
        self.assertTrue(samples[0][0])
        means = g.infer_mean(1)
        self.assertEqual(len(means), 1, "exactly one node queried")

    def test_beta(self):
        g = graph.Graph()
        c1 = g.add_constant_pos_real(1.1)
        c2 = g.add_constant_pos_real(5.0)
        # negative tests on number of parents
        # 0 parents not allowed
        with self.assertRaises(ValueError) as cm:
            g.add_distribution(
                graph.DistributionType.BETA, graph.AtomicType.PROBABILITY, []
            )
        self.assertTrue(
            "Beta distribution must have exactly two parents" in str(cm.exception)
        )
        # 1 parent not allowed
        with self.assertRaises(ValueError) as cm:
            g.add_distribution(
                graph.DistributionType.BETA, graph.AtomicType.PROBABILITY, [c1]
            )
        self.assertTrue(
            "Beta distribution must have exactly two parents" in str(cm.exception)
        )
        # negative test on type of parent
        c3 = g.add_constant(True)
        with self.assertRaises(ValueError) as cm:
            g.add_distribution(
                graph.DistributionType.BETA, graph.AtomicType.PROBABILITY, [c3, c3]
            )
        self.assertTrue("must be positive real-valued" in str(cm.exception))
        # negative test on sample type
        with self.assertRaises(ValueError) as cm:
            g.add_distribution(
                graph.DistributionType.BETA, graph.AtomicType.REAL, [c1, c2]
            )
        self.assertTrue("Beta produces probability samples" in str(cm.exception))
        # 2 real-valued parents with probability sample type are OK
        d1 = g.add_distribution(
            graph.DistributionType.BETA, graph.AtomicType.PROBABILITY, [c1, c2]
        )
        # now let's draw some samples from the Beta distribution
        v1 = g.add_operator(graph.OperatorType.SAMPLE, [d1])
        g.query(v1)
        samples = g.infer(1, graph.InferenceType.REJECTION)
        self.assertEqual(type(samples[0][0]), float)
        self.assertTrue(samples[0][0] > 0 and samples[0][0] < 1)
        means = g.infer_mean(10000, graph.InferenceType.REJECTION)
        self.assertAlmostEqual(means[0], 1.1 / (1.1 + 5.0), 2, "beta mean")

    def test_binomial(self):
        g = graph.Graph()
        c1 = g.add_constant(10)
        c2 = g.add_constant_probability(0.55)
        d1 = g.add_distribution(
            graph.DistributionType.BINOMIAL, graph.AtomicType.NATURAL, [c1, c2]
        )
        v1 = g.add_operator(graph.OperatorType.SAMPLE, [d1])
        g.query(v1)
        samples = g.infer(1, graph.InferenceType.REJECTION)
        self.assertEqual(type(samples[0][0]), int)
        self.assertTrue(samples[0][0] <= 10)
        means = g.infer_mean(10000, graph.InferenceType.REJECTION)
        self.assertTrue(means[0] > 5 and means[0] < 6)

    def _create_graph(self):
        g = graph.Graph()
        c1 = g.add_constant_col_simplex_matrix(np.array([0.8, 0.2]))
        c2 = g.add_constant_col_simplex_matrix(np.array([[0.6, 0.99], [0.4, 0.01]]))
        c3 = g.add_constant_col_simplex_matrix(
            np.transpose(np.array([[1, 0], [0.2, 0.8], [0.1, 0.9], [0.01, 0.99]]))
        )
        Rain = g.add_operator(
            graph.OperatorType.SAMPLE,
            [
                g.add_distribution(
                    graph.DistributionType.TABULAR, graph.AtomicType.BOOLEAN, [c1]
                )
            ],
        )
        Sprinkler = g.add_operator(
            graph.OperatorType.SAMPLE,
            [
                g.add_distribution(
                    graph.DistributionType.TABULAR, graph.AtomicType.BOOLEAN, [c2, Rain]
                )
            ],
        )
        GrassWet = g.add_operator(
            graph.OperatorType.SAMPLE,
            [
                g.add_distribution(
                    graph.DistributionType.TABULAR,
                    graph.AtomicType.BOOLEAN,
                    [c3, Sprinkler, Rain],
                )
            ],
        )
        return g, Rain, Sprinkler, GrassWet

    def test_query(self):
        g, Rain, Sprinkler, GrassWet = self._create_graph()
        g.query(Rain)
        g.query(Sprinkler)
        g.query(GrassWet)
        g.infer(1)

    def test_to_dot(self):
        self.maxDiff = None
        g, Rain, Sprinkler, GrassWet = self._create_graph()
        g.query(Rain)
        g.query(Sprinkler)
        g.query(GrassWet)
        g.observe(GrassWet, True)
        observed = g.to_dot()
        expected = """
digraph "graph" {
  N0[label="simplex"];
  N1[label="simplex"];
  N2[label="simplex"];
  N3[label="Tabular"];
  N4[label="~"];
  N5[label="Tabular"];
  N6[label="~"];
  N7[label="Tabular"];
  N8[label="~"];
  N0 -> N3;
  N1 -> N5;
  N2 -> N7;
  N3 -> N4;
  N4 -> N5;
  N4 -> N7;
  N5 -> N6;
  N6 -> N7;
  N7 -> N8;
  O0[label="Observation"];
  N8 -> O0;
  Q0[label="Query"];
  N4 -> Q0;
  Q1[label="Query"];
  N6 -> Q1;
  Q2[label="Query"];
  N8 -> Q2;
}"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_observe(self):
        g, Rain, Sprinkler, GrassWet = self._create_graph()
        g.observe(GrassWet, True)
        with self.assertRaises(ValueError) as cm:
            g.observe(GrassWet, True)
        self.assertTrue("duplicate observe for node" in str(cm.exception))

        g = graph.Graph()
        c1 = g.add_constant_probability(1.0)
        c2 = g.add_constant_probability(0.5)
        o1 = g.add_operator(graph.OperatorType.MULTIPLY, [c1, c2])
        d1 = g.add_distribution(
            graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, [o1]
        )
        o2 = g.add_operator(graph.OperatorType.SAMPLE, [d1])
        with self.assertRaises(ValueError) as cm:
            g.observe(o1, True)
        self.assertTrue("only sample nodes may be observed" in str(cm.exception))
        g.observe(o2, True)  # ok to observe this node
        with self.assertRaises(ValueError) as cm:
            g.observe(o2, False)
        self.assertTrue("duplicate observe" in str(cm.exception))
        g.remove_observations()
        g.observe(o2, False)

    def test_inference(self):
        g, Rain, Sprinkler, GrassWet = self._create_graph()
        g.observe(GrassWet, True)
        g.query(Rain)
        g.query(GrassWet)
        with self.assertRaises(ValueError) as cm:
            g.query(Rain)
        self.assertTrue("duplicate query for node" in str(cm.exception))
        samples = g.infer(1)
        self.assertTrue(len(samples) == 1)
        # since we have observed grass wet is true the query should be true
        self.assertEqual(type(samples[0][1]), bool)
        self.assertTrue(samples[0][1])
        # test parallel inference
        samples_all = g.infer(num_samples=1, n_chains=2)
        self.assertTrue(len(samples_all) == 2)
        self.assertTrue(len(samples_all[0]) == 1)
        self.assertTrue(len(samples_all[1]) == 1)
        self.assertEqual(samples[0][0], samples_all[0][0][0])
        self.assertEqual(samples[0][1], samples_all[0][0][1])
        self.assertEqual(type(samples_all[1][0][0]), bool)
        self.assertEqual(type(samples_all[1][0][1]), bool)
        self.assertTrue(samples_all[1][0][1])

    def test_infer_mean(self):
        g = graph.Graph()
        c1 = g.add_constant_probability(1.0)
        op1 = g.add_operator(graph.OperatorType.MULTIPLY, [c1, c1])
        d1 = g.add_distribution(
            graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, [op1]
        )
        op2 = g.add_operator(graph.OperatorType.SAMPLE, [d1])
        g.query(op1)
        g.query(op2)
        means = g.infer_mean(100)
        self.assertAlmostEqual(means[0], 1.0)
        self.assertAlmostEqual(means[1], 1.0)
        # test parallel inference
        means_all = g.infer_mean(num_samples=100, n_chains=2)
        self.assertTrue(len(means_all) == 2)
        self.assertAlmostEqual(means_all[0][0], 1.0)
        self.assertAlmostEqual(means_all[0][1], 1.0)
        self.assertAlmostEqual(means_all[1][0], 1.0)
        self.assertAlmostEqual(means_all[1][1], 1.0)

    def test_neg_real(self):
        g = graph.Graph()
        with self.assertRaises(ValueError) as cm:
            g.add_constant_neg_real(1.25)
        self.assertTrue("neg_real must be <=0" in str(cm.exception))
        neg1 = g.add_constant_neg_real(-1.25)
        expected = """
        Node 0 type 1 parents [ ] children [ ] negative real -1.25
        """
        self.assertEqual(g.to_string().strip(), expected.strip())
        add_negs = g.add_operator(graph.OperatorType.ADD, [neg1, neg1])
        g.query(add_negs)
        means = g.infer_mean(10)
        self.assertAlmostEqual(means[0], -2.5)
        samples = g.infer(10)
        self.assertAlmostEqual(samples[0][0], -2.5)

    def test_get_log_prob(self):
        g, Rain, Sprinkler, GrassWet = self._create_graph()
        g.observe(GrassWet, True)
        g.query(Rain)
        g.query(GrassWet)
        conf = graph.InferConfig()
        conf.keep_log_prob = True
        g.infer(
            num_samples=10,
            algorithm=graph.InferenceType.GIBBS,
            seed=123,
            n_chains=2,
            infer_config=conf,
        )
        log_probs = g.get_log_prob()
        self.assertEqual(len(log_probs), 2)
        self.assertEqual(len(log_probs[0]), 10)
