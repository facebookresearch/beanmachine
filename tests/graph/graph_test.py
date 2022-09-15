# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
            [c2, g.add_constant_bool(True)],
        )

        with self.assertRaises(ValueError) as cm:
            g.add_distribution(
                graph.DistributionType.TABULAR,
                graph.AtomicType.BOOLEAN,
                [c2, g.add_constant_natural(1)],
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
        c3 = g.add_constant_natural(1)
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
        c3 = g.add_constant_bool(True)
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
        c1 = g.add_constant_natural(10)
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

    def test_categorical(self):
        g = graph.Graph()
        simplex = [0.5, 0.25, 0.125, 0.125]
        c1 = g.add_constant_col_simplex_matrix(np.array(simplex))
        # Negative test: Number of parents must be exactly one:
        with self.assertRaises(ValueError) as cm:
            g.add_distribution(
                graph.DistributionType.CATEGORICAL, graph.AtomicType.NATURAL, []
            )
        self.assertTrue(
            "Categorical distribution must have exactly one parent" in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            g.add_distribution(
                graph.DistributionType.CATEGORICAL, graph.AtomicType.NATURAL, [c1, c1]
            )
        self.assertEqual(
            "Categorical distribution must have exactly one parent", str(cm.exception)
        )

        # Negative test: parent must be simplex:
        c3 = g.add_constant_natural(1)
        with self.assertRaises(ValueError) as cm:
            g.add_distribution(
                graph.DistributionType.CATEGORICAL, graph.AtomicType.NATURAL, [c3]
            )
        self.assertEqual(
            "Categorical parent must be a one-column simplex", str(cm.exception)
        )

        # Negative test: type must be natural
        with self.assertRaises(ValueError) as cm:
            g.add_distribution(
                graph.DistributionType.CATEGORICAL, graph.AtomicType.REAL, [c1]
            )
        self.assertEqual(
            "Categorical produces natural valued samples", str(cm.exception)
        )

        # Positive test:
        d1 = g.add_distribution(
            graph.DistributionType.CATEGORICAL, graph.AtomicType.NATURAL, [c1]
        )

        v1 = g.add_operator(graph.OperatorType.SAMPLE, [d1])
        g.query(v1)
        num_samples = 10000
        # TODO: We use rejection sampling in this test because at present NMC
        # does not support inference over naturals. If inference over discrete
        # variables is important for BMG, we should create a Uniform Proposer
        # similar to how it's done in Bean Machine proper.
        samples = g.infer(
            num_samples=num_samples,
            algorithm=graph.InferenceType.REJECTION,
            seed=123,
            n_chains=1,
        )[0]

        # The distribution of the samples should closely match the simplex used to
        # generate them.

        histogram = [0, 0, 0, 0]
        for sample in samples:
            histogram[sample[0]] += 1

        self.assertAlmostEqual(simplex[0], histogram[0] / num_samples, delta=0.01)
        self.assertAlmostEqual(simplex[1], histogram[1] / num_samples, delta=0.01)
        self.assertAlmostEqual(simplex[2], histogram[2] / num_samples, delta=0.01)
        self.assertAlmostEqual(simplex[3], histogram[3] / num_samples, delta=0.01)

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

        p = g.add_constant_probability(0.8)
        b = g.add_distribution(
            graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, [p]
        )
        # Querying a constant is weird but allowed
        g.query(p)
        # But querying a distribution directly rather than a sample is
        # illegal:
        with self.assertRaises(ValueError) as cm:
            g.query(b)
        self.assertEqual(
            f"Query of node_id {b} expected a node of type 1 or 3 but is 2",
            str(cm.exception),
        )

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
        self.assertTrue(
            "only SAMPLE and IID_SAMPLE nodes may be observed" in str(cm.exception)
        )
        g.observe(o2, True)  # ok to observe this node
        with self.assertRaises(ValueError) as cm:
            g.observe(o2, False)
        self.assertTrue("duplicate observe" in str(cm.exception))
        g.remove_observations()
        g.observe(o2, False)

    def test_inference(self):
        g, Rain, Sprinkler, GrassWet = self._create_graph()
        g.observe(GrassWet, True)
        qr = g.query(Rain)
        g.query(GrassWet)
        # Querying the same node twice is idempotent.
        self.assertEqual(g.query(Rain), qr)
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

    def test_graph_stats(self):
        g = graph.Graph()
        c1 = g.add_constant_natural(10)
        c2 = g.add_constant_probability(0.55)
        d1 = g.add_distribution(
            graph.DistributionType.BINOMIAL, graph.AtomicType.NATURAL, [c1, c2]
        )
        g.add_operator(graph.OperatorType.SAMPLE, [d1])
        stats = g.collect_statistics()
        self.maxDiff = None
        expected = """
Graph Statistics Report
#######################
Number of nodes: 4
Number of edges: 3
Graph density: 0.25
Number of root nodes: 2
Number of terminal nodes: 1
Maximum no. of incoming edges into a node: 2
Maximum no. of outgoing edges from a node: 1

Node statistics:
################
CONSTANT: 2
\tRoot nodes: 2
\tConstant node statistics:
\t-------------------------
\t\tPROBABILITY and SCALAR: 1
\t\tNATURAL and SCALAR: 1

\t\tDistribution of incoming edges:
\t\t-------------------------------
\t\tNodes with 0 edges: 2

\t\tDistribution of outgoing edges:
\t\t-------------------------------
\t\tNodes with 1 edges: 2

DISTRIBUTION: 1
\tNo root or terminal nodes
\tDistribution node statistics:
\t-----------------------------
\t\tBINOMIAL: 1

\t\tDistribution of incoming edges:
\t\t-------------------------------
\t\tNodes with 2 edges: 1

\t\tDistribution of outgoing edges:
\t\t-------------------------------
\t\tNodes with 1 edges: 1

OPERATOR: 1
\tTerminal nodes: 1
\tOperator node statistics:
\t-------------------------
\t\tSAMPLE: 1

\t\tDistribution of incoming edges:
\t\t-------------------------------
\t\tNodes with 1 edges: 1

\t\tDistribution of outgoing edges:
\t\t-------------------------------
\t\tNodes with 0 edges: 1

Edge statistics:
################
\tDistribution of incoming edges:
\t-------------------------------
\tNodes with 0 edges: 2
\tNodes with 1 edges: 1
\tNodes with 2 edges: 1

\tDistribution of outgoing edges:
\t-------------------------------
\tNodes with 0 edges: 1
\tNodes with 1 edges: 3

"""
        self.assertEqual(stats.strip(), expected.strip())


class TestContinuousModels(unittest.TestCase):
    def test_product_distribution(self):
        g = graph.Graph()

        MEAN0 = -5.0
        STD0 = 1.0
        real0 = g.add_constant(MEAN0)
        pos0 = g.add_constant_pos_real(STD0)

        normal_dist0 = g.add_distribution(
            graph.DistributionType.NORMAL, graph.AtomicType.REAL, [real0, pos0]
        )

        real1 = g.add_operator(graph.OperatorType.SAMPLE, [normal_dist0])

        STD1 = 2.0
        pos1 = g.add_constant_pos_real(STD1)

        normal_dist1 = g.add_distribution(
            graph.DistributionType.NORMAL, graph.AtomicType.REAL, [real1, pos1]
        )

        MEAN2 = 5.0
        STD2 = 2.0
        real2 = g.add_constant(MEAN2)
        pos2 = g.add_constant_pos_real(STD2)

        normal_dist2 = g.add_distribution(
            graph.DistributionType.NORMAL, graph.AtomicType.REAL, [real2, pos2]
        )

        product_dist1 = g.add_distribution(
            graph.DistributionType.PRODUCT,
            graph.AtomicType.REAL,
            [normal_dist1, normal_dist2],
        )

        product_sample1 = g.add_operator(graph.OperatorType.SAMPLE, [product_dist1])

        product_sample2 = g.add_operator(graph.OperatorType.SAMPLE, [product_dist1])

        product_sample3 = g.add_operator(graph.OperatorType.SAMPLE, [product_dist1])

        g.observe(product_sample1, -1.0)
        g.observe(product_sample2, 0.0)
        g.observe(product_sample3, 1.0)

        g.query(real1)

        default_config = graph.InferConfig()
        samples = g.infer(
            num_samples=10000,
            algorithm=graph.InferenceType.NMC,
            seed=5123401,
            n_chains=1,
            infer_config=default_config,
        )
        chain = 0
        variable = 0
        values = [sample_tuple[variable] for sample_tuple in samples[chain]]
        mean = sum(values) / len(values)
        print(mean)
        expected = -2.848  # obtained from the same test ran in C++
        self.assertAlmostEqual(mean, expected, delta=0.1)
