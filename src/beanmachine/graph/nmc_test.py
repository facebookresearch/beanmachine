# Copyright (c) Facebook, Inc. and its affiliates.
import math
import unittest

import numpy as np
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
        # for each school we will add two random variables,
        # but first we need to define a distribution
        school_effect_dist = g.add_distribution(
            graph.DistributionType.NORMAL,
            graph.AtomicType.REAL,
            [overall_mean, overall_std],
        )
        for treatment_mean_value, treatment_std_value in DATA:
            # school_effect ~ Normal(overall_mean, overall_std)
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

    # see https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case
    # we are assuming zero mean here for simplicity
    def test_bivariate_gaussian(self):
        g = graph.Graph()
        flat = g.add_distribution(
            graph.DistributionType.FLAT, graph.AtomicType.REAL, []
        )
        x = g.add_operator(graph.OperatorType.SAMPLE, [flat])
        y = g.add_operator(graph.OperatorType.SAMPLE, [flat])
        x_sq = g.add_operator(graph.OperatorType.MULTIPLY, [x, x])
        y_sq = g.add_operator(graph.OperatorType.MULTIPLY, [y, y])
        x_y = g.add_operator(graph.OperatorType.MULTIPLY, [x, y])
        SIGMA_X = 5.0
        SIGMA_Y = 2.0
        RHO = 0.7
        x_sq_term = g.add_constant(-0.5 / (1 - RHO ** 2) / SIGMA_X ** 2)
        g.add_factor(graph.FactorType.EXP_PRODUCT, [x_sq, x_sq_term])
        y_sq_term = g.add_constant(-0.5 / (1 - RHO ** 2) / SIGMA_Y ** 2)
        g.add_factor(graph.FactorType.EXP_PRODUCT, [y_sq, y_sq_term])
        x_y_term = g.add_constant(RHO / (1 - RHO ** 2) / SIGMA_X / SIGMA_Y)
        g.add_factor(graph.FactorType.EXP_PRODUCT, [x_y, x_y_term])
        g.query(x)
        g.query(x_sq)
        g.query(y)
        g.query(y_sq)
        g.query(x_y)
        means = g.infer_mean(10000, graph.InferenceType.NMC)
        print("means", means)  # only printed on error
        self.assertTrue(abs(means[0] - 0.0) < 0.2, "mean of x should be 0")
        self.assertTrue(
            abs(means[1] - SIGMA_X ** 2) < 0.2, f"mean of x^2 should be {SIGMA_X**2}"
        )
        self.assertTrue(abs(means[2] - 0.0) < 0.2, "mean of y should be 0")
        self.assertTrue(
            abs(means[3] - SIGMA_Y ** 2) < 0.2, f"mean of y^2 should be {SIGMA_Y**2}"
        )
        post_cov = means[4] / math.sqrt(means[1]) / math.sqrt(means[3])
        self.assertTrue(
            abs(post_cov - RHO) < 0.2, f"covariance should be {RHO} is {post_cov}"
        )

    def test_probit_regression(self):
        """
        x ~ Normal(0, 1)
        y ~ Bernoulli(Phi(x))
        P(Phi(x) | y = true) ~ Beta(2, 1)
        """
        g = graph.Graph()
        zero = g.add_constant(0.0)
        one = g.add_constant_pos_real(1.0)
        prior = g.add_distribution(
            graph.DistributionType.NORMAL, graph.AtomicType.REAL, [zero, one]
        )
        x = g.add_operator(graph.OperatorType.SAMPLE, [prior])
        phi_x = g.add_operator(graph.OperatorType.PHI, [x])
        likelihood = g.add_distribution(
            graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, [phi_x]
        )
        y = g.add_operator(graph.OperatorType.SAMPLE, [likelihood])
        g.observe(y, True)
        phi_x_sq = g.add_operator(graph.OperatorType.MULTIPLY, [phi_x, phi_x])
        g.query(phi_x)
        g.query(phi_x_sq)
        means = g.infer_mean(10000, graph.InferenceType.NMC)
        post_var = means[1] - means[0] ** 2
        self.assertAlmostEquals(
            means[0], 2 / (2 + 1), 2, f"posterior mean {means[0]} is not accurate"
        )
        self.assertAlmostEquals(
            post_var,
            2 * 1 / (2 + 1) ** 2 / (2 + 1 + 1),
            2,
            f"posterior variance {post_var} is not accurate",
        )

    def test_clara_gp(self):
        """
        CLARA-GP model
        f() ~ GP(0, squared_exp_covar)
        for each labeler l:
            spec_l ~ Beta(SPEC_ALPHA, SPEC_BETA)
            sens_l ~ Beta(SENS_ALPHA, SENS_BETA)
        for each item i
            violating_i ~ Bernoulli(Phi(f(i)))
            for each labeler l
                if violating_i
                    prob_i_l = sens_l
                else
                    prob_i_l = 1 - spec_l
                label_i_l ~ Bernoulli(prob_i_l)
        """
        ALPHA = 1.0
        RHO = 0.1
        SENS_ALPHA = 9.0
        SENS_BETA = 1.0
        SPEC_ALPHA = 9.5
        SPEC_BETA = 0.5
        NUM_LABELERS = 2
        SCORES = np.array([0.1, 0.2, 0.3])
        ITEM_LABELS = [[False, False], [False, True], [True, True]]
        # see https://mc-stan.org/docs/2_19/functions-reference/covariance.html for
        # a reference on this covariance function
        covar = ALPHA ** 2 * np.exp(
            -((np.expand_dims(SCORES, 1) - SCORES) ** 2) / 2 / RHO ** 2
        )
        tau = np.linalg.inv(covar)  # the precision matrix
        g = graph.Graph()
        # first we will create f ~ GP
        flat = g.add_distribution(
            graph.DistributionType.FLAT, graph.AtomicType.REAL, []
        )
        f = [g.add_operator(graph.OperatorType.SAMPLE, [flat]) for _ in SCORES]
        for i in range(len(SCORES)):
            tau_i_i = g.add_constant(-0.5 * tau[i, i])
            g.add_factor(graph.FactorType.EXP_PRODUCT, [tau_i_i, f[i], f[i]])
            for j in range(i + 1, len(SCORES)):
                tau_i_j = g.add_constant(-1.0 * tau[i, j])
                g.add_factor(graph.FactorType.EXP_PRODUCT, [tau_i_j, f[i], f[j]])
        # for each labeler l:
        #     spec_l ~ Beta(SPEC_ALPHA, SPEC_BETA)
        #     sens_l ~ Beta(SENS_ALPHA, SENS_BETA)
        spec_alpha = g.add_constant_pos_real(SPEC_ALPHA)
        spec_beta = g.add_constant_pos_real(SPEC_BETA)
        spec_prior = g.add_distribution(
            graph.DistributionType.BETA,
            graph.AtomicType.PROBABILITY,
            [spec_alpha, spec_beta],
        )
        sens_alpha = g.add_constant_pos_real(SENS_ALPHA)
        sens_beta = g.add_constant_pos_real(SENS_BETA)
        sens_prior = g.add_distribution(
            graph.DistributionType.BETA,
            graph.AtomicType.PROBABILITY,
            [sens_alpha, sens_beta],
        )
        spec, comp_spec, sens = [], [], []
        for labeler in range(NUM_LABELERS):
            spec.append(g.add_operator(graph.OperatorType.SAMPLE, [spec_prior]))
            comp_spec.append(
                g.add_operator(graph.OperatorType.COMPLEMENT, [spec[labeler]])
            )
            sens.append(g.add_operator(graph.OperatorType.SAMPLE, [sens_prior]))
        # for each item i
        for i, labels in enumerate(ITEM_LABELS):
            # violating_i ~ Bernoulli(Phi(f(i)))
            dist_i = g.add_distribution(
                graph.DistributionType.BERNOULLI,
                graph.AtomicType.BOOLEAN,
                [g.add_operator(graph.OperatorType.PHI, [f[i]])],
            )
            violating_i = g.add_operator(graph.OperatorType.SAMPLE, [dist_i])
            # for each labeler l
            for lidx, label_val in enumerate(labels):
                # if violating_i
                #     prob_i_l = sens_l
                # else
                #     prob_i_l = 1 - spec_l
                prob_i_l = g.add_operator(
                    graph.OperatorType.IF_THEN_ELSE,
                    [violating_i, sens[lidx], comp_spec[lidx]],
                )
                # label_i_l ~ Bernoulli(prob_i_l)
                dist_i_l = g.add_distribution(
                    graph.DistributionType.BERNOULLI,
                    graph.AtomicType.BOOLEAN,
                    [prob_i_l],
                )
                label_i_l = g.add_operator(graph.OperatorType.SAMPLE, [dist_i_l])
                g.observe(label_i_l, label_val)
            g.query(violating_i)
        means = g.infer_mean(1000, graph.InferenceType.NMC)
        self.assertLess(means[0], means[1])
        self.assertLess(means[1], means[2])

    def test_uncoupled_bools(self):
        """
        X_1 ~ Bernoulli(0.5)
        X_2 ~ Bernoulli(0.5)
        P(X_1 == X_2) = 0.5
        """
        g = graph.Graph()
        half = g.add_constant_probability(0.5)
        bernoulli = g.add_distribution(
            graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, [half]
        )
        X_1 = g.add_operator(graph.OperatorType.SAMPLE, [bernoulli])
        X_2 = g.add_operator(graph.OperatorType.SAMPLE, [bernoulli])
        g.query(X_1)
        g.query(X_2)
        prob_equal = (
            sum(x == y for (x, y) in g.infer(100000, graph.InferenceType.NMC)) / 100000
        )
        self.assertAlmostEqual(prob_equal, 0.5, delta=0.01)

    def test_coupled_bools(self):
        """
        X_1 ~ Bernoulli(0.5)
        X_2 ~ Bernoulli(0.5)
        sigma_1 = 1 if X_1 else -1
        sigma_2 = 1 if X_2 else -1
        target += sigma_1 * sigma_2
        P(X_1 == X_2) = e / (e + e^-1)
        """
        g = graph.Graph()
        half = g.add_constant_probability(0.5)
        bernoulli = g.add_distribution(
            graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, [half]
        )
        X_1 = g.add_operator(graph.OperatorType.SAMPLE, [bernoulli])
        X_2 = g.add_operator(graph.OperatorType.SAMPLE, [bernoulli])
        plus_one = g.add_constant(1.0)
        minus_one = g.add_constant(-1.0)
        sigma_1 = g.add_operator(
            graph.OperatorType.IF_THEN_ELSE, [X_1, plus_one, minus_one]
        )
        sigma_2 = g.add_operator(
            graph.OperatorType.IF_THEN_ELSE, [X_2, plus_one, minus_one]
        )
        g.add_factor(graph.FactorType.EXP_PRODUCT, [sigma_1, sigma_2])
        g.query(X_1)
        g.query(X_2)
        prob_equal = (
            sum(x == y for (x, y) in g.infer(100000, graph.InferenceType.NMC)) / 100000
        )
        self.assertAlmostEqual(prob_equal, 0.88, delta=0.01)

    @classmethod
    def create_GPfactor(cls, bmg, alpha, rho, scores, mu=0.0):
        # see https://mc-stan.org/docs/2_19/functions-reference/covariance.html for
        # a reference on this covariance function
        covar = alpha ** 2 * np.exp(
            -((np.expand_dims(scores, 1) - scores) ** 2) / 2 / rho ** 2
        )
        tau = np.linalg.inv(covar)  # the precision matrix
        neg_mu = bmg.add_constant(-mu)
        # f ~ GP
        flat = bmg.add_distribution(
            graph.DistributionType.FLAT, graph.AtomicType.REAL, []
        )
        f = [bmg.add_operator(graph.OperatorType.SAMPLE, [flat]) for _ in scores]
        if mu == 0.0:
            f_centered = f
        else:
            f_centered = [
                bmg.add_operator(graph.OperatorType.ADD, [fi, neg_mu]) for fi in f
            ]
        for i in range(len(scores)):
            tau_i_i = bmg.add_constant(-0.5 * tau[i, i])
            bmg.add_factor(
                graph.FactorType.EXP_PRODUCT, [tau_i_i, f_centered[i], f_centered[i]]
            )
            for j in range(i + 1, len(scores)):
                tau_i_j = bmg.add_constant(-1.0 * tau[i, j])
                bmg.add_factor(
                    graph.FactorType.EXP_PRODUCT,
                    [tau_i_j, f_centered[i], f_centered[j]],
                )
        return f

    @classmethod
    def sum_negate_nodes(cls, bmg, in_nodes):
        result = bmg.add_operator(
            graph.OperatorType.NEGATE,
            [
                bmg.add_operator(
                    graph.OperatorType.TO_REAL,
                    [bmg.add_operator(graph.OperatorType.ADD, in_nodes)],
                )
            ],
        )
        return result

    def test_clara_gp_logit(self):
        """
        CLARA-GP model with prev, sens, spec in logit space
        f_prev() ~ GP(0, squared_exp_covar)
        f_sens() ~ GP(logit(0.9), squared_exp_covar)
        f_spec() ~ GP(logit(0.95), squared_exp_covar)
        for each item i
            log_prev_i = -log1pexp(-f_prev(i)) # log(prev_i)
            log_comp_prev_i = -log1pexp(f_prev(i)) # log(1 - prev_i)
            # assume all labeller share the same sens and spec
            # so sens and spec only depends on score, indexed by i
            log_spec_i = -log1pexp(-f_spec(i))
            log_com_spec_i = -log1pexp(f_spec(i))
            log_sens_i = -log1pexp(-f_sens(i))
            log_comp_sens_i = -log1pexp(f_sens(i))
            loglik1, loglik2 = log_prev_i, log_comp_prev_i
            for each label
                loglik1 += label_i_l ? log_sens_i : log_comp_sens_i
                loglik2 += label_i_l ? log_comp_spec_i : log_spec_i
            add factor:
                logsumexp(loglik1, loglk2)
        """
        ALPHA = 1.0
        RHO = 0.1
        SPEC_MU = 2.9  # logit(0.95)
        SENS_MU = 2.2  # logit(0.9)
        # NUM_LABELERS = 2
        SCORES = np.array([0.1, 0.2, 0.3])
        ITEM_LABELS = [[False, False], [False, True], [True, True]]
        # create f ~ GP
        g = graph.Graph()
        f_prev = self.create_GPfactor(g, ALPHA, RHO, SCORES)
        f_spec = self.create_GPfactor(g, ALPHA, RHO, SCORES, SPEC_MU)
        f_sens = self.create_GPfactor(g, ALPHA, RHO, SCORES, SENS_MU)
        # for each factor:
        #   -log(p) = lop1pexp(-f)
        #   -log(1-p) = log1pexp(f)
        # note: the followings log_* are negative log probabilities,
        #   negate right before LOGSUMEXP

        # for each item i
        for i, labels in enumerate(ITEM_LABELS):
            # in this test case, we assume labelers share the same spec and sens
            log_spec = g.add_operator(
                graph.OperatorType.LOG1PEXP,
                [g.add_operator(graph.OperatorType.NEGATE, [f_spec[i]])],
            )
            log_comp_spec = g.add_operator(graph.OperatorType.LOG1PEXP, [f_spec[i]])
            log_sens = g.add_operator(
                graph.OperatorType.LOG1PEXP,
                [g.add_operator(graph.OperatorType.NEGATE, [f_sens[i]])],
            )
            log_comp_sens = g.add_operator(graph.OperatorType.LOG1PEXP, [f_sens[i]])
            log_prev = g.add_operator(
                graph.OperatorType.LOG1PEXP,
                [g.add_operator(graph.OperatorType.NEGATE, [f_prev[i]])],
            )
            log_comp_prev = g.add_operator(graph.OperatorType.LOG1PEXP, [f_prev[i]])
            loglik1, loglik2 = [log_prev], [log_comp_prev]
            # for each labeler l
            for label_val in labels:
                if label_val:
                    loglik1.append(log_sens)
                    loglik2.append(log_comp_spec)
                else:
                    loglik1.append(log_comp_sens)
                    loglik2.append(log_spec)
            loglik1 = self.sum_negate_nodes(g, loglik1)
            loglik2 = self.sum_negate_nodes(g, loglik2)
            g.add_factor(
                graph.FactorType.EXP_PRODUCT,
                [g.add_operator(graph.OperatorType.LOGSUMEXP, [loglik1, loglik2])],
            )
            g.query(f_prev[i])
        means = g.infer_mean(1000, graph.InferenceType.NMC)
        self.assertLess(means[0], means[1])
        self.assertLess(means[1], means[2])
