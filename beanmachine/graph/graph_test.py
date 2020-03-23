# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torch
from beanmachine import graph


class TestBayesNet(unittest.TestCase):
    def test_simple_dep(self):
        g = graph.Graph()
        c1 = g.add_constant(torch.FloatTensor([0.8, 0.2]))
        d1 = g.add_distribution(
            graph.DistributionType.TABULAR, graph.AtomicType.BOOLEAN, [c1]
        )
        g.add_operator(graph.OperatorType.SAMPLE, [d1])

    def test_tabular(self):
        g = graph.Graph()
        c1 = g.add_constant(torch.FloatTensor([0.8, 0.2]))

        # negative test
        with self.assertRaises(ValueError) as cm:
            g.add_distribution(
                graph.DistributionType.TABULAR, graph.AtomicType.BOOLEAN, []
            )
        self.assertTrue(
            "Tabular distribution first arg must be tensor" in str(cm.exception)
        )

        g = graph.Graph()
        c1 = g.add_constant(torch.FloatTensor([0.8, 0.2]))
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
        self.assertTrue("expected 3 dims got 1" in str(cm.exception))

        c2 = g.add_constant(torch.FloatTensor([[0.6, 0.4], [0.99, 0.01]]))
        g.add_distribution(
            graph.DistributionType.TABULAR,
            graph.AtomicType.BOOLEAN,
            [c2, g.add_constant(True)],
        )

        with self.assertRaises(ValueError) as cm:
            g.add_distribution(
                graph.DistributionType.TABULAR,
                graph.AtomicType.BOOLEAN,
                [c2, g.add_constant(torch.tensor(1))],
            )
        self.assertTrue("only supports boolean parents" in str(cm.exception))

        c3 = g.add_constant(torch.FloatTensor([1.1, -0.1]))
        with self.assertRaises(ValueError) as cm:
            g.add_distribution(
                graph.DistributionType.TABULAR, graph.AtomicType.BOOLEAN, [c3]
            )
        self.assertTrue("must be positive" in str(cm.exception))

        c4 = g.add_constant(torch.FloatTensor([0.9, 0.0999]))
        with self.assertRaises(ValueError) as cm:
            g.add_distribution(
                graph.DistributionType.TABULAR, graph.AtomicType.BOOLEAN, [c4]
            )
        self.assertTrue("must add to 1" in str(cm.exception))

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
        c3 = g.add_constant(torch.scalar_tensor(1, dtype=torch.int))
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
        self.assertTrue(samples[0][0].type == graph.AtomicType.BOOLEAN)
        self.assertTrue(samples[0][0].bool)
        means = g.infer_mean(1)
        self.assertEqual(len(means), 1, "exactly one node queried")

    def test_beta(self):
        g = graph.Graph()
        c1 = g.add_constant(1.1)
        c2 = g.add_constant(5.0)
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
        self.assertTrue("must be real-valued" in str(cm.exception))
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
        self.assertTrue(samples[0][0].type == graph.AtomicType.PROBABILITY)
        self.assertTrue(samples[0][0].probability > 0 and samples[0][0].probability < 1)
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
        self.assertTrue(samples[0][0].type == graph.AtomicType.NATURAL)
        self.assertTrue(samples[0][0].natural <= 10)
        means = g.infer_mean(10000, graph.InferenceType.REJECTION)
        self.assertTrue(means[0] > 5 and means[0] < 6)

    def _create_graph(self):
        g = graph.Graph()
        c1 = g.add_constant(torch.FloatTensor([0.8, 0.2]))
        c2 = g.add_constant(torch.FloatTensor([[0.6, 0.4], [0.99, 0.01]]))
        c3 = g.add_constant(
            torch.FloatTensor([[[1, 0], [0.2, 0.8]], [[0.1, 0.9], [0.01, 0.99]]])
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
        self.assertTrue(samples[0][1].type == graph.AtomicType.BOOLEAN)
        self.assertTrue(samples[0][1].bool)

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
        # negative test, don't support aggregating tensors
        c2 = g.add_constant(torch.tensor(0.5))
        op2 = g.add_operator(graph.OperatorType.ADD, [c2, c2])
        g.query(op2)
        with self.assertRaises(RuntimeError):
            g.infer_mean(100)
        g.infer(100)  # infer should be fine though
