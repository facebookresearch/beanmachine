# Copyright (c) Facebook, Inc. and its affiliates.
# """End-to-end test for n-schools model based on the one in PPL Bench"""
# See for example https://github.com/facebookresearch/pplbench/blob/master/pplbench/models/n_schools.py

import logging
import unittest
from typing import Tuple

import beanmachine.ppl as bm
import numpy as np
import torch.distributions as dist
import xarray as xr
from beanmachine import graph
from beanmachine.ppl.inference.bmg_inference import BMGInference
from scipy.stats import norm
from torch import tensor

LOGGER = logging.getLogger(__name__)

# Planned additions:

# TODO: It would be great to have another example based on Sepehr's bma model here:
# https://www.internalfb.com/intern/diffusion/FBS/browsefile/master/fbcode/beanmachine/beanmachine/applications/fb/bma/bma_model.py

# TODO: It would be great to also have a test case based on the PPL bench version of n-schools that follows:
# Start n-schools model
"""
N Schools
This is a generalization of a classical 8 schools model to n schools.
The model posits that the effect of a school on a student's performance
can be explained by the a baseline effect of all schools plus an additive
effect of the state, the school district and the school type.
Hyper Parameters:
    n - total number of schools
    num_states - number of states
    num_districts_per_state - number of school districts in each state
    num_types - number of school types
    scale_state - state effect scale
    scale_district - district effect scale
    scale_type - school type effect scale
Model:
    beta_baseline = StudentT(dof_baseline, 0.0, scale_baseline)
    sigma_state ~ HalfCauchy(0, scale_state)
    sigma_district ~ HalfCauchy(0, scale_district)
    sigma_type ~ HalfCauchy(0, scale_type)
    for s in 0 .. num_states - 1
        beta_state[s] ~ Normal(0, sigma_state)
        for d in 0 .. num_districts_per_state - 1
            beta_district[s, d] ~ Normal(0, sigma_district)
    for t in 0 .. num_types - 1
        beta_type[t] ~ Normal(0, sigma_type)
    for i in 0 ... n - 1
        Assume we are given state[i], district[i], type[i]
        Y_hat[i] = beta_baseline + beta_state[state[i]]
                    + beta_district[state[i], district[i]]
                    + beta_type[type[i]]
        sigma[i] ~ Uniform(0.5, 1.5)
        Y[i] ~ Normal(Y_hat[i], sigma[i])
The dataset consists of the following
    Y[school]         - float
    sigma[school]     - float
and it includes the attributes
    n  - number of schools
    num_states
    num_districts_per_state
    num_types
    dof_baseline
    scale_baseline
    scale_state
    scale_district
    scale_type
    state_idx[school]     - 0 .. num_states - 1
    district_idx[school]  - 0 .. num_districts_per_state - 1
    type_idx[school]      - 0 .. num_types - 1
The posterior samples include the following,
    sigma_state[draw]                    - float
    sigma_district[draw]                 - float
    sigma_type[draw]                     - float
    beta_baseline[draw]                  - float
    beta_state[draw, state]              - float
    beta_district[draw, state, district] - float
    beta_type[draw, type]                - float
"""


def generate_data(  # type: ignore
    seed: int,
    n: int = 2000,
    num_states: int = 8,
    num_districts_per_state: int = 5,
    num_types: int = 5,
    dof_baseline: float = 3.0,
    scale_baseline: float = 10.0,
    scale_state: float = 1.0,
    scale_district: float = 1.0,
    scale_type: float = 1.0,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    See the class documentation for an explanation of the parameters.
    :param seed: random number generator seed
    """
    if n % 2 != 0:
        LOGGER.warn(f"n should be a multiple of 2. Actual values = {n}")
    # In this model we will generate exactly equal amounts of training
    # and test data with the same number of training and test schools
    # in each state, district, and type combination
    n = n // 2
    rng = np.random.default_rng(seed)
    beta_baseline = rng.standard_t(dof_baseline) * scale_baseline
    sigma_state = np.abs(rng.standard_cauchy()) * scale_state
    sigma_district = np.abs(rng.standard_cauchy()) * scale_district
    sigma_type = np.abs(rng.standard_cauchy()) * scale_type
    beta_state = rng.normal(loc=0, scale=sigma_state, size=num_states)
    beta_district = rng.normal(
        loc=0, scale=sigma_district, size=(num_states, num_districts_per_state)
    )
    beta_type = rng.normal(loc=0, scale=sigma_type, size=num_types)

    # we will randomly assign the schools to states, district, and types
    state_idx = rng.integers(low=0, high=num_states, size=n)
    district_idx = rng.integers(low=0, high=num_districts_per_state, size=n)
    type_idx = rng.integers(low=0, high=num_types, size=n)

    y_hat = (
        beta_baseline
        + beta_state[state_idx]
        + beta_district[state_idx, district_idx]
        + beta_type[type_idx]
    )
    train_sigma = rng.uniform(0.5, 1.5, size=n)
    train_y = rng.normal(loc=y_hat, scale=train_sigma)
    test_sigma = rng.uniform(0.5, 1.5, size=n)
    test_y = rng.normal(loc=y_hat, scale=test_sigma)

    return tuple(  # type: ignore
        xr.Dataset(
            {"Y": (["school"], y), "sigma": (["school"], sigma)},
            coords={"school": np.arange(n)},
            attrs={
                "n": n,
                "num_states": num_states,
                "num_districts_per_state": num_districts_per_state,
                "num_types": num_types,
                "dof_baseline": dof_baseline,
                "scale_baseline": scale_baseline,
                "scale_state": scale_state,
                "scale_district": scale_district,
                "scale_type": scale_type,
                "state_idx": state_idx,
                "district_idx": district_idx,
                "type_idx": type_idx,
            },
        )
        for y, sigma in [(train_y, train_sigma), (test_y, test_sigma)]
    )


def evaluate_posterior_predictive(samples: xr.Dataset, test: xr.Dataset) -> np.ndarray:
    """
    Computes the predictive likelihood of all the test items w.r.t. each sample.
    See the class documentation for the `samples` and `test` parameters.
    :returns: a numpy array of the same size as the sample dimension.
    """
    # transpose the datasets to be in a convenient format
    samples = samples.transpose("draw", "state", "district", "type")
    y_hat = (
        samples.beta_baseline.values[:, np.newaxis]
        + samples.beta_state.values[:, test.attrs["state_idx"]]
        + samples.beta_district.values[
            :, test.attrs["state_idx"], test.attrs["district_idx"]
        ]
        + samples.beta_type.values[:, test.attrs["type_idx"]]
    )  # size = (iterations, n_test)
    loglike = norm.logpdf(
        test.Y.values[np.newaxis, :],
        loc=y_hat,
        scale=test.sigma.values[np.newaxis, :],
    )  # size = (iterations, n_test)
    return loglike.sum(axis=1)  # size = (iterations,)


## OLD

# DATA
NUM_CLASS = 2  # num_classes (For Dirichlet may be need at least two)
# TODO: Clarify number of lablers is implicit
NUM_ITEMS = 1  # number of items
PREV_PRIOR = tensor([1.0, 0.0])  # prior on prevalence
# PREV_PRIOR is a list of length NUM_CLASSES
CONF_MATRIX_PRIOR = tensor([1.0, 0.0])  # prior on confusion matrix
# CONF_MATRIX_PRIOR is a list of length NUM_CLASS
# TODO: Does Dirichlet support 2d matrices?
# TODO: Is it really necessary to reject dirichlet on tensor([1])?
IDX_RATINGS = [[0]]  # indexed ratings that labelers assigned to items
IDX_LABELERS = [[0]]  # indexed list of labelers who labeled items
EXPERT_CONF_MATRIX = tensor(
    [[0.99, 0.01], [0.01, 0.99]]
)  # confusion matrix of an expert (if we have true ratings)
# EXPERT_CONF_MATRIX is of size NUM_CLASS x NUM_CLASS
# Row (first) index is true class, and column (second) index is observed
IDX_TRUE_RATINGS = [0]
# Of size NUM_ITEMS
# Represents true class of items by a perfect labler
# When information is missing, use value NUM_CLASS

# MODEL


@bm.random_variable
def prevalence():
    # Dirichlet distribution support is implemented in Beanstalk but not yet landed.
    return dist.Dirichlet(PREV_PRIOR)


@bm.random_variable
def confusion_matrix(labeler, true_class):
    return dist.Dirichlet(CONF_MATRIX_PRIOR)  # size: NUM_CLASSES


# log of the unnormalized item probs
# log P(true label of item i = k | labels)
# shape: [NUM_ITEMS, NUM_CLASSES]
@bm.functional
def log_item_prob(i, k):
    # Indexing into a simplex with a constant is implemented
    # but not yet landed
    prob = prevalence()[k].log()
    for r in range(len(IDX_RATINGS[i])):
        label = IDX_RATINGS[i][r]
        labeler = IDX_LABELERS[i][r]
        prob = prob + confusion_matrix(labeler, k)[label].log()
    if IDX_TRUE_RATINGS[i] != NUM_CLASS:  # Value NUM_CLASS means missing value
        prob = prob + EXPERT_CONF_MATRIX[k, IDX_TRUE_RATINGS[i]].log()
    return prob


# log of joint prob of labels, prev, conf_matrix
@bm.random_variable
def target():
    joint_log_prob = 0
    for i in range(NUM_ITEMS):
        # logsumexp on a newly-constructed tensor with stochastic
        # elements has limited support but this should work:
        log_probs = tensor(
            # TODO: Hard-coded k in {0,1}
            # [log_item_prob(i, 0), log_item_prob(i, 1), log_item_prob(i, 2)]
            # [log_item_prob(i, 0), log_item_prob(i, 1)]
            [log_item_prob(i, k) for k in range(NUM_CLASS)]
        )
        joint_log_prob = joint_log_prob + log_probs.logsumexp(0)
    return dist.Bernoulli(joint_log_prob.exp())


observations = {target(): tensor(1.0)}
queries = [
    log_item_prob(0, 0),  # Ideally, all the other elements too
    prevalence(),
    confusion_matrix(0, 0),  # Ideally, all the other elements too
]
ssrw = "SingleSiteRandomWalk"
bmgi = "BMG inference"
both = {ssrw, bmgi}
# TODO: Replace 4th param of expecteds by more methodical calculation
expecteds = [
    (prevalence(), both, 0.5000, 0.001),
    (confusion_matrix(0, 0), both, 0.5000, 0.001),
    (log_item_prob(0, 0), {ssrw}, -1.3863, 0.5),
    (log_item_prob(0, 0), {bmgi}, -1.0391, 0.5),
]


class NSchoolsTest(unittest.TestCase):
    def test_eight_schools_e2e(self):
        # see https://www.jstatsoft.org/article/view/v012i03/v12i03.pdf
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
        observed = g.to_dot()
        expected = """
digraph "graph" {
  N0[label="0"];
  N1[label="1000"];
  N2[label="Normal"];
  N3[label="~"];
  N4[label="HalfCauchy"];
  N5[label="~"];
  N6[label="Normal"];
  N7[label="~"];
  N8[label="14.9"];
  N9[label="Normal"];
  N10[label="~"];
  N11[label="~"];
  N12[label="10.2"];
  N13[label="Normal"];
  N14[label="~"];
  N15[label="~"];
  N16[label="16.3"];
  N17[label="Normal"];
  N18[label="~"];
  N19[label="~"];
  N20[label="11"];
  N21[label="Normal"];
  N22[label="~"];
  N23[label="~"];
  N24[label="9.4"];
  N25[label="Normal"];
  N26[label="~"];
  N27[label="~"];
  N28[label="11.4"];
  N29[label="Normal"];
  N30[label="~"];
  N31[label="~"];
  N32[label="10.4"];
  N33[label="Normal"];
  N34[label="~"];
  N35[label="~"];
  N36[label="17.6"];
  N37[label="Normal"];
  N38[label="~"];
  N0 -> N2;
  N1 -> N2;
  N1 -> N4;
  N2 -> N3;
  N3 -> N6;
  N4 -> N5;
  N5 -> N6;
  N6 -> N7;
  N6 -> N11;
  N6 -> N15;
  N6 -> N19;
  N6 -> N23;
  N6 -> N27;
  N6 -> N31;
  N6 -> N35;
  N7 -> N9;
  N8 -> N9;
  N9 -> N10;
  N11 -> N13;
  N12 -> N13;
  N13 -> N14;
  N15 -> N17;
  N16 -> N17;
  N17 -> N18;
  N19 -> N21;
  N20 -> N21;
  N21 -> N22;
  N23 -> N25;
  N24 -> N25;
  N25 -> N26;
  N27 -> N29;
  N28 -> N29;
  N29 -> N30;
  N31 -> N33;
  N32 -> N33;
  N33 -> N34;
  N35 -> N37;
  N36 -> N37;
  N37 -> N38;
  O0[label="Observation"];
  N10 -> O0;
  O1[label="Observation"];
  N14 -> O1;
  O2[label="Observation"];
  N18 -> O2;
  O3[label="Observation"];
  N22 -> O3;
  O4[label="Observation"];
  N26 -> O4;
  O5[label="Observation"];
  N30 -> O5;
  O6[label="Observation"];
  N34 -> O6;
  O7[label="Observation"];
  N38 -> O7;
  Q0[label="Query"];
  N7 -> Q0;
  Q1[label="Query"];
  N11 -> Q1;
  Q2[label="Query"];
  N15 -> Q2;
  Q3[label="Query"];
  N19 -> Q3;
  Q4[label="Query"];
  N23 -> Q4;
  Q5[label="Query"];
  N27 -> Q5;
  Q6[label="Query"];
  N31 -> Q6;
  Q7[label="Query"];
  N35 -> Q7;
  Q8[label="Query"];
  N3 -> Q8;
  Q9[label="Query"];
  N5 -> Q9;
}"""
        self.assertTrue(expected, observed)

        means = g.infer_mean(1000, graph.InferenceType.NMC)
        for idx, (mean, std) in enumerate(EXPECTED):
            self.assertTrue(
                abs(means[idx] - mean) < std * 0.5,
                f"index {idx} expected {mean} +- {std*0.5} actual {means[idx]}",
            )

    # TODO: The following tests should be turned into working tests focused on
    # n-schools (rather than the CLARA examples they are templated on.)
    def disabled_test_nschools_tensor_cgm_no_update_inference(self) -> None:
        """Check BM and BMG inference both terminate"""

        self.maxDiff = None
        num_samples = 10

        # First, let's see how the model fairs with Random Walk inference
        inference = bm.SingleSiteRandomWalk()  # or NUTS
        mcsamples = inference.infer(queries, observations, num_samples)

        for rand_var, inferences, value, delta in expecteds:
            if ssrw in inferences:
                samples = mcsamples[rand_var]
                observed = samples.mean()
                expected = tensor([value])
                self.assertAlmostEqual(first=observed, second=expected, delta=delta)

        # Second, let's see how it fairs with the bmg inference
        inference = BMGInference()
        mcsamples = inference.infer(queries, observations, num_samples)

        for rand_var, inferences, value, delta in expecteds:
            if bmgi in inferences:
                samples = mcsamples[rand_var]
                observed = samples.mean()
                expected = tensor([value])
                self.assertAlmostEqual(first=observed, second=expected, delta=delta)

    def disabled_test_nschools_tensor_cgm_no_update_to_dot_cpp_python(self) -> None:
        self.maxDiff = None
        observed = BMGInference().to_dot(queries, observations)
        expected = """
digraph "graph" {
  N00[label="[1.0,0.0]"];
  N01[label=Dirichlet];
  N02[label=Sample];
  N03[label=Sample];
  N04[label=Sample];
  N05[label=0];
  N06[label=index];
  N07[label=Log];
  N08[label=index];
  N09[label=Log];
  N10[label="+"];
  N11[label=-0.010050326585769653];
  N12[label="+"];
  N13[label=1];
  N14[label=index];
  N15[label=Log];
  N16[label=index];
  N17[label=Log];
  N18[label="+"];
  N19[label=-4.605170249938965];
  N20[label="+"];
  N21[label=LogSumExp];
  N22[label=ToReal];
  N23[label=Exp];
  N24[label=ToProb];
  N25[label=Bernoulli];
  N26[label=Sample];
  N27[label="Observation True"];
  N28[label=Query];
  N29[label=Query];
  N30[label=Query];
  N00 -> N01;
  N01 -> N02;
  N01 -> N03;
  N01 -> N04;
  N02 -> N06;
  N02 -> N14;
  N02 -> N29;
  N03 -> N08;
  N03 -> N30;
  N04 -> N16;
  N05 -> N06;
  N05 -> N08;
  N05 -> N16;
  N06 -> N07;
  N07 -> N10;
  N08 -> N09;
  N09 -> N10;
  N10 -> N12;
  N11 -> N12;
  N12 -> N21;
  N12 -> N28;
  N13 -> N14;
  N14 -> N15;
  N15 -> N18;
  N16 -> N17;
  N17 -> N18;
  N18 -> N20;
  N19 -> N20;
  N20 -> N21;
  N21 -> N22;
  N22 -> N23;
  N23 -> N24;
  N24 -> N25;
  N25 -> N26;
  N26 -> N27;
}
        """
        self.assertEqual(observed.strip(), expected.strip())

        observed = BMGInference().to_cpp(queries, observations)
        expected = """
graph::Graph g;
Eigen::MatrixXd m0(2, 1)
m0 << 1.0, 0.0;
uint n0 = g.add_constant_pos_matrix(m0);
uint n1 = g.add_distribution(
  graph::DistributionType::DIRICHLET,
  graph::ValueType(
    graph::VariableType::COL_SIMPLEX_MATRIX,
    graph::AtomicType::PROBABILITY,
    2,
    1
  )
  std::vector<uint>({n0}));
uint n2 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n3 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n4 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n5 = g.add_constant(0);
uint n6 = g.add_operator(
  graph::OperatorType::INDEX, std::vector<uint>({n2, n5}));
uint n7 = g.add_operator(
  graph::OperatorType::LOG, std::vector<uint>({n6}));
uint n8 = g.add_operator(
  graph::OperatorType::INDEX, std::vector<uint>({n3, n5}));
uint n9 = g.add_operator(
  graph::OperatorType::LOG, std::vector<uint>({n8}));
uint n10 = g.add_operator(
  graph::OperatorType::ADD, std::vector<uint>({n7, n9}));
uint n11 = g.add_constant_neg_real(-0.010050326585769653);
uint n12 = g.add_operator(
  graph::OperatorType::ADD, std::vector<uint>({n10, n11}));
uint n13 = g.add_constant(1);
uint n14 = g.add_operator(
  graph::OperatorType::INDEX, std::vector<uint>({n2, n13}));
uint n15 = g.add_operator(
  graph::OperatorType::LOG, std::vector<uint>({n14}));
uint n16 = g.add_operator(
  graph::OperatorType::INDEX, std::vector<uint>({n4, n5}));
uint n17 = g.add_operator(
  graph::OperatorType::LOG, std::vector<uint>({n16}));
uint n18 = g.add_operator(
  graph::OperatorType::ADD, std::vector<uint>({n15, n17}));
uint n19 = g.add_constant_neg_real(-4.605170249938965);
uint n20 = g.add_operator(
  graph::OperatorType::ADD, std::vector<uint>({n18, n19}));
n21 = g.add_operator(
  graph::OperatorType::LOGSUMEXP,
  std::vector<uint>({n12, n20}));
uint n22 = g.add_operator(
  graph::OperatorType::TO_REAL, std::vector<uint>({n21}));
uint n23 = g.add_operator(
  graph::OperatorType::EXP, std::vector<uint>({n22}));
uint n24 = g.add_operator(
  graph::OperatorType::TO_PROBABILITY, std::vector<uint>({n23}));
uint n25 = g.add_distribution(
  graph::DistributionType::BERNOULLI,
  graph::AtomicType::BOOLEAN,
  std::vector<uint>({n24}));
uint n26 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n25}));
g.observe([n26], true);
g.query(n12);
g.query(n2);
g.query(n3);
"""
        self.assertEqual(observed.strip(), expected.strip())

        observed = BMGInference().to_python(queries, observations)
        expected = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant_pos_matrix(tensor([[1.0],[0.0]]))
n1 = g.add_distribution(
  graph.DistributionType.DIRICHLET,
  graph.ValueType(
    graph.VariableType.COL_SIMPLEX_MATRIX,
    graph.AtomicType.PROBABILITY,
    2,
    1,
  ),
  [n0],
)
n2 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n3 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n4 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n5 = g.add_constant(0)
n6 = g.add_operator(graph.OperatorType.INDEX, [n2, n5])
n7 = g.add_operator(graph.OperatorType.LOG, [n6])
n8 = g.add_operator(graph.OperatorType.INDEX, [n3, n5])
n9 = g.add_operator(graph.OperatorType.LOG, [n8])
n10 = g.add_operator(graph.OperatorType.ADD, [n7, n9])
n11 = g.add_constant_neg_real(-0.010050326585769653)
n12 = g.add_operator(graph.OperatorType.ADD, [n10, n11])
n13 = g.add_constant(1)
n14 = g.add_operator(graph.OperatorType.INDEX, [n2, n13])
n15 = g.add_operator(graph.OperatorType.LOG, [n14])
n16 = g.add_operator(graph.OperatorType.INDEX, [n4, n5])
n17 = g.add_operator(graph.OperatorType.LOG, [n16])
n18 = g.add_operator(graph.OperatorType.ADD, [n15, n17])
n19 = g.add_constant_neg_real(-4.605170249938965)
n20 = g.add_operator(graph.OperatorType.ADD, [n18, n19])
n21 = g.add_operator(
  graph.OperatorType.LOGSUMEXP,
  [n12, n20])
n22 = g.add_operator(graph.OperatorType.TO_REAL, [n21])
n23 = g.add_operator(graph.OperatorType.EXP, [n22])
n24 = g.add_operator(graph.OperatorType.TO_PROBABILITY, [n23])
n25 = g.add_distribution(
  graph.DistributionType.BERNOULLI,
  graph.AtomicType.BOOLEAN,
  [n24])
n26 = g.add_operator(graph.OperatorType.SAMPLE, [n25])
g.observe(n26, True)
g.query(n12)
g.query(n2)
g.query(n3)
"""
        self.assertEqual(observed.strip(), expected.strip())
