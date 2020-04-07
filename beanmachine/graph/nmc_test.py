# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

from beanmachine import graph


class TestNMC(unittest.TestCase):
    # see https://www.jstatsoft.org/article/view/v012i03/v12i03.pdf
    def test_eight_schools(self):
        # For each school, the average treatment effect and the standard deviation
        DATA = [
            (28.39, 14.9),
            (7.94, 10.2),
            (-2.75, 16.3),
            (6.82, 11.0),
            (-0.64, 9.4),
            (0.63, 11.4),
            (18.01, 10.4),
            (12.16, 17.6),
        ]
        # the expected mean and standard deviation of each random variable
        EXPECTED = [
            (11.1, 9.1),
            (7.6, 6.6),
            (5.7, 8.4),
            (7.1, 7.0),
            (5.1, 6.8),
            (5.7, 7.3),
            (10.4, 7.3),
            (8.3, 8.4),
            (7.6, 5.9),  # overall mean
            (6.7, 5.6),  # overall std
        ]
        g = graph.Graph()
        zero = g.add_constant(0.0)
        thousand = g.add_constant_pos_real(1000.0)
        # overall_mean ~ Normal(0, 1000)
        overall_mean_dist = g.add_distribution(
            graph.DistributionType.NORMAL, graph.AtomicType.REAL, [zero, thousand]
        )
        overall_mean = g.add_operator(graph.OperatorType.SAMPLE, [overall_mean_dist])
        # overall_std ~ HalfCauchy(1000)
        # [note: the original paper had overall_std ~ Uniform(0, 1000)]
        overall_std_dist = g.add_distribution(
            graph.DistributionType.HALF_CAUCHY, graph.AtomicType.POS_REAL, [thousand]
        )
        overall_std = g.add_operator(graph.OperatorType.SAMPLE, [overall_std_dist])
        # for each school we will add two random variables
        for treatment_mean_value, treatment_std_value in DATA:
            # school_effect ~ Normal(overall_mean, overall_std)
            school_effect_dist = g.add_distribution(
                graph.DistributionType.NORMAL,
                graph.AtomicType.REAL,
                [overall_mean, overall_std],
            )
            school_effect = g.add_operator(
                graph.OperatorType.SAMPLE, [school_effect_dist]
            )
            g.query(school_effect)
            # treatment_mean ~ Normal(school_effect, treatment_std)
            treatment_std = g.add_constant_pos_real(treatment_std_value)
            treatment_mean_dist = g.add_distribution(
                graph.DistributionType.NORMAL,
                graph.AtomicType.REAL,
                [school_effect, treatment_std],
            )
            treatment_mean = g.add_operator(
                graph.OperatorType.SAMPLE, [treatment_mean_dist]
            )
            g.observe(treatment_mean, treatment_mean_value)
        g.query(overall_mean)
        g.query(overall_std)
        means = g.infer_mean(1000, graph.InferenceType.NMC)
        for idx, (mean, std) in enumerate(EXPECTED):
            self.assertTrue(
                abs(means[idx] - mean) < std * 0.5,
                f"index {idx} expected {mean} +- {std*0.5} actual {means[idx]}",
            )
