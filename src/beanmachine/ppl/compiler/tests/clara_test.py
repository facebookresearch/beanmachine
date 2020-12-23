#!/usr/bin/env python3

# Non-vectorized CLARA model with probabilities represented
# as straight probabilities, not negative reals.

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import tensor
from torch.distributions import Bernoulli, Normal


n_prev_buckets = 2  # a positive integer
n_sens_buckets = 2  # a positive integer
n_spec_buckets = 2  # a positive integer
num_items = 2  # a positive integer

# TODO: Get this working as in the quip
# TODO: Make a version of this that actually makes sense: that
# TODO: uses real data structures rather than treating everything as a tensor
# TODO: or list.

# TODO: Say more clearly what these represent; we are choosing
# TODO: a row from the L_K arrays below.
# n_prev_buckets positive integers each from 0 -> n_prev_buckets-1
prev_indices = [0, 1]

# n_sens_buckets positive integers each from 0 -> n_sens_buckets-1
sens_indices = [1, 1]

# n_spec_buckets positive integers each from 0 -> n_spec_buckets-1
spec_indices = [1, 0]

# n_prev_buckets x n_prev_buckets positive integers
# each from 0 -> n_prev_buckets-1
L_K_prev = tensor([[0, 1], [1, 1]])

# n_sens_buckets x n_sens_buckets positive integers
# each from 0 -> n_sens_buckets-1
L_K_sens = tensor([[0, 1], [1, 1]])

# n_spec_buckets x n_spec_buckets positive integers
# each from 0 -> n_spec_buckets-1
L_K_spec = tensor([[0, 1], [1, 1]])

prev_f_mean = 1.2  #  A real
sens_f_mean = 2.3  #  A real
spec_f_mean = 3.4  #  A real


# num_items positive integers
# number of occurrences of each kind of label.
#
n_labels = [3, 1]
# Booleans; number of bools is sum(n_labels)
labels = [True, True, False, False]

# TODO: What does this represent?
# num_items positive integers
n_repeats = [2, 3]


# Takes a real, produces a probability
phi = Normal(0.0, 1.0).cdf


# Takes a positive integer up to n_prev_buckets
# Produces a real
# TODO: Say what these represent
@bm.random_variable
def eta_prev(n):
    return Normal(0.0, 1.0)


# Takes a positive integer up to n_sens_buckets
# Produces a real
@bm.random_variable
def eta_sens(n):
    return Normal(0.0, 1.0)


# Takes a positive integer up to n_spec_buckets
# Produces a real
@bm.random_variable
def eta_spec(n):
    return Normal(0.0, 1.0)


# Produces a bool, which we observe to be true.
@bm.random_variable
def observation():
    idx = 0  # current index into labels
    prob = 1  # prob
    for i in range(num_items):  # item index
        prev_idx = prev_indices[i]  # 0 -> n_prev_buckets - 1
        sens_idx = sens_indices[i]  # 0 -> n_sens_buckets - 1
        spec_idx = spec_indices[i]  # 0 -> n_spec_buckets - 1

        prev = prev_f_mean  # real
        for ii in range(n_prev_buckets):  # ii is positive integer
            # Every quantity here is real
            prev = prev + L_K_prev[prev_idx, ii] * eta_prev(ii)
        prob_prev = phi(prev)  # prob

        spec = spec_f_mean  # real
        for ii in range(0, n_spec_buckets):  # ii pos int
            # Sum of reals
            spec = spec + L_K_spec[spec_idx, ii] * eta_spec(ii)
        # PROBLEM: We know that phi(spec) is a probability, but
        # we have no way of knowing that 0.48 * phi(spec) + 0.5 is
        # a probability.
        prob_spec = 0.48 * phi(spec) + 0.5  # pos real, should be prob

        sens = sens_f_mean
        for ii in range(n_sens_buckets):
            sens = sens + L_K_sens[sens_idx, ii] * eta_sens(ii)
        prob_sens = 0.48 * phi(sens) + 0.5  # pos real, should be prob

        pos_sum = prob_prev  # prob
        neg_sum = 1 - pos_sum  # prob

        for j in range(n_labels[i]):  # j is positive integer
            label = labels[idx + j]  # label is Boolean
            if label:
                # TODO: Implement *=, +=, etc, in the compiler
                pos_sum = pos_sum * prob_sens  # pos real or real, should be prob
                neg_sum = neg_sum * (1 - prob_spec)  # real, should be prob
            else:
                pos_sum = pos_sum * (1 - prob_sens)  # real, should be prob
                neg_sum = neg_sum * prob_spec  # pos real or real, should be prob

        idx = idx + n_labels[i]

        # PROBLEM: Even if we know that pos_sum and neg_sum
        # are probs, how do we know that the sum of them
        # is a probability? What reason do we have to believe
        # that this sum is between 0.0 and 1.0?

        # (I know it is true and you know it is true -- it is
        # true because they *started* as summing to 1.0 and
        # we have only made them both *smaller* so their sum
        # must be smaller than 1.0. But how does BMG know that?)

        prob_i = (pos_sum + neg_sum) ** n_repeats[i]
        prob = prob * prob_i

    return Bernoulli(prob)


class ClaraTest(unittest.TestCase):
    def test_clara_inference(self) -> None:
        self.maxDiff = None
        queries = [
            eta_prev(0),
            eta_prev(1),
            eta_spec(0),
            eta_spec(1),
            eta_sens(0),
            eta_sens(1),
        ]
        observations = {observation(): tensor(1.0)}
        num_samples = 1000
        inference = BMGInference()
        # TODO: Right now this gives an error because (as noted above)
        # we do not know that the sum of two reals is a probability.
        # Until we've figured out how to fix that, note that this
        # should produce an exception.
        # TODO: Have this throw a better exception than ValueError.
        with self.assertRaises(ValueError):
            inference.infer(queries, observations, num_samples)
