#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.compiler.hint import log1mexp
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import tensor


"""
In this version of the CLARA model we change the generative
process of each labeler's confusion matrix. For each row in
the labeler's confusion matrix:

diagonal_element ~ 1 - 0.5 * Beta(a, b)
other_elements   ~ (1 - diagonal_element) * Dirichlet(K-1)

This way, diagonal_element + other_elements = 1
but we have forced diagonal_element > 0.5.
"""


NUM_LABELS = 3
BRONZE = 0
SILVER = 1
GOLD = 2

NUM_LABELERS = 3
SUE = 0
BOB = 1
EVE = 2

NUM_ITEMS = 4

# The labels given to items by labelers; a ragged array with
# NUM_ITEMS rows. Each row has no more than NUM_LABELERS labels.
ITEM_LABELS = [
    [GOLD, GOLD, SILVER],
    [BRONZE, SILVER],
    [SILVER, SILVER],
    [BRONZE, BRONZE],
]

# The labelers who labeled the items; must have exact same shape as ITEM_LABELS.
IDX_LABELERS = [
    [SUE, BOB, EVE],
    [SUE, EVE],
    [SUE, BOB],
    [BOB, EVE],
]

# The expert confusion matrix is a NUM_LABELS x NUM_LABELS matrix giving the
# probability of an expert producing each possible label for an item with a given
# true label. The first row for instance is "if an expert is given an item that
# should be labeled BRONZE, there is a 90% chance that the item is labeled BRONZE,
# 7% SILVER, 3% GOLD"
EXPERT_CONF_MATRIX = tensor(
    [
        [0.9, 0.7, 0.3],
        [0.5, 0.9, 0.5],
        [0.3, 0.7, 0.9],
    ]
)


# The true label of each item, or -1 if the true label is unknown.
TRUE_LABELS = [GOLD, SILVER, -1, BRONZE]


# Produces a simplex of length NUM_LABELS.
# Each entry is the probability that it is the true label of random item.
@bm.random_variable
def prevalence():
    PREVALENCE_PRIOR = torch.ones(NUM_LABELS)
    return dist.Dirichlet(PREVALENCE_PRIOR)


# Used to compute the probability that the labeler *correctly* labels an
# item with the given true label. In the next method we force this
# quantity to be > 0.5.
@bm.random_variable
def confusion_diag(labeler, true_label):
    SOME_CONSTANT1 = 2.0
    SOME_CONSTANT2 = 2.0
    return dist.Beta(SOME_CONSTANT1, SOME_CONSTANT2)


# Force the probability of a correct label to be >0.5 and
# take its log. Note that this is stable because the operand
# of the log is not close to zero.
@bm.functional
def log_constrained_confusion_diag(labeler, true_label):
    return torch.log(1 - 0.5 * confusion_diag(labeler, true_label))


# This is the log-probability of getting *any* incorrect label; it is
# the inverse of the probability of a correct label. Again, this computation
# is stable even if the probability of getting an incorrect label is
# close to zero.
@bm.functional
def log1m_constrained_confusion_diag(labeler, true_label):
    # TODO: Fix the compiler so that we recognize log(1-exp(x)) and replace
    # it automatically with a LOG1MEXP node rather than relying on a call
    # to a hint helper.
    return log1mexp(log_constrained_confusion_diag(labeler, true_label))


# Produces a simplex of length NUM_LABELS-1.
# This is used to compute probability that the labeler *incorrectly* labels an
# item with the given true label, so there are NUM_LABELS-1 possibilities.
@bm.random_variable
def confusion_non_diag(labeler, true_label):
    return dist.Dirichlet(torch.ones(NUM_LABELS - 1))


# Produces a NUM_LABELS x NUM_LABELS matrix which answers the question
# "For an item with a given true label, what is the probability that
# this labeler will assign each possible label?"
@bm.functional
def log_confusion_matrix(labeler):
    # Original code was
    #
    # log_conf_matrix = torch.ones(NUM_LABELS, NUM_LABELS)
    #
    # but we do not support mutation of a tensor with stochastic graph
    # elements in BMG.  Instead, construct a list, mutate that, and
    # then turn it into a tensor.

    log_conf_matrix = [[None] * NUM_LABELS for _ in range(NUM_LABELS)]
    for true_label in range(NUM_LABELS):
        # Start by filling in the diagonal; the diagonal is the case where the
        # labeler gets it right.
        log_conf_matrix[true_label][true_label] = log_constrained_confusion_diag(
            labeler, true_label
        )
        # Now fill in all the cases where the labeler gets it wrong.
        for observed_label in range(true_label):
            log_conf_matrix[true_label][
                observed_label
            ] = log1m_constrained_confusion_diag(labeler, true_label) + torch.log(
                confusion_non_diag(labeler, true_label)[observed_label]
            )
        for observed_label in range(true_label + 1, NUM_LABELS):
            log_conf_matrix[true_label][
                observed_label
            ] = log1m_constrained_confusion_diag(labeler, true_label) + torch.log(
                confusion_non_diag(labeler, true_label)[observed_label - 1]
            )
    return tensor(log_conf_matrix)


@bm.functional
def log_item_prob(item, true_label):
    prob = torch.log(prevalence()[true_label])
    for label_index in range(len(ITEM_LABELS[item])):
        label = ITEM_LABELS[item][label_index]
        labeler = IDX_LABELERS[item][label_index]
        # TODO: The more natural way to write this would be [true_label, label]
        # but we do not support stochastic tuple indices yet.
        # TODO: The compiler still does not handle models that contain +=.
        prob = prob + log_confusion_matrix(labeler)[true_label][label]
    if TRUE_LABELS[item] != -1:
        # TODO: Similarly with the indexing here.
        prob = prob + torch.log(EXPERT_CONF_MATRIX[true_label][TRUE_LABELS[item]])
    return prob


# log of joint prob of labels, prevalence, confusion matrix
@bm.random_variable
def target(item):
    all_item_probs = [
        log_item_prob(item, true_label) for true_label in range(NUM_LABELS)
    ]
    joint_log_prob = torch.logsumexp(tensor(all_item_probs), dim=0)
    return dist.Bernoulli(torch.exp(joint_log_prob))


class Clara2Test(unittest.TestCase):
    def test_clara2_inference(self) -> None:
        self.maxDiff = None

        observations = {target(item): tensor(1.0) for item in range(NUM_ITEMS)}
        # Given observations of labeler's choice of label for each item we wish to know:
        # * for every item, what is the probability of each possible label being correct?
        # * what is the true prevalence of each label?
        # * what are the confusion matrices for each labeler?
        queries = (
            [
                log_item_prob(item, true_label)
                for item in range(NUM_ITEMS)
                for true_label in range(NUM_LABELS)
            ]
            + [prevalence()]
            + [log_confusion_matrix(labeler) for labeler in range(NUM_LABELERS)]
        )
        results = BMGInference().infer(queries, observations, 100)
        item_0_bronze = results[log_item_prob(0, BRONZE)].mean()
        item_0_silver = results[log_item_prob(0, SILVER)].mean()
        item_0_gold = results[log_item_prob(0, GOLD)].mean()
        cm_sue = results[log_confusion_matrix(0)].exp().mean(dim=1)

        # These are non-normalized log-probabilities that item 0 is of each
        # possible label; the softmax function would normalize them.  Note
        # that the item is judged to be much more likely to be gold than silver,
        # and much more likely to be silver than bronze.
        self.assertAlmostEqual(first=-9.75, second=item_0_bronze, delta=1.0)
        self.assertAlmostEqual(first=-7.48, second=item_0_silver, delta=1.0)
        self.assertAlmostEqual(first=-4.13, second=item_0_gold, delta=1.0)

        # The confusion matrix gives the probability that labeler Sue assigns
        # a given label to an item. For example, we infer that when given an item
        # whose true label is bronze, Sue assigns bronze 76% of the time,
        # silver 11% of the time and gold 13% of the time.

        self.assertAlmostEqual(first=0.76, second=cm_sue[0, BRONZE, BRONZE], delta=0.1)
        self.assertAlmostEqual(first=0.11, second=cm_sue[0, BRONZE, SILVER], delta=0.1)
        self.assertAlmostEqual(first=0.13, second=cm_sue[0, BRONZE, GOLD], delta=0.1)
