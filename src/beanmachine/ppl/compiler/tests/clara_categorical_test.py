#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

# This is a simplified version of the CLARA model which uses categorical
# distributions. We do not support compiling this to BMG yet because
# we do not have categorical or Dirichlet distributions in BMG or
# stochastic control flows other than if-then-else.
#
# This module tests whether we can (1) accumulate a graph for this
# model, and (2) produce sensible error messages.

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import tensor
from torch.distributions import Categorical, Dirichlet


# three categories
sheep = 0
goat = 1
cat = 2

# two classifiers
alice = 10
bob = 11

# two photos
foo_jpg = 20
bar_jpg = 21


@bm.random_variable
def distribution():
    # Flat prior on true distribution of categories amongst documents:
    # no distribution is more likely than any other.
    return Dirichlet(tensor([1.0, 1.0, 1.0]))


# Prior on confusion of classifiers; classifiers confuse sheep with
# goats more easily than sheep with cats or goats with cats:
confusion = [
    # Given a sheep, we are unlikely to classify it as a cat
    tensor([10.0, 5.0, 1.0]),
    # Given a goat, we are unlikely to classify it as a cat
    tensor([5.0, 10.0, 1.0]),
    # Given a cat, we are likely to classify it as a cat
    tensor([1.0, 1.0, 10.0]),
]

# For each classifier and category, sample a categorical distribution
# of likely classifications of an item truly of the category.
@bm.random_variable
def theta(classifier: int, category: int):
    return Dirichlet(confusion[category])


# For each item, sample from a collection of items whose true
# categories are distributed by pi. The output sample is a category.
@bm.random_variable
def category_of_item(item: int):
    return Categorical(distribution())


# We then simulate classification of item by classifier.
# Given true category z(item) for the item, the behaviour of
# classifier will be to sample from a categorical distribution
# whose parameters are theta(j, z(item))
# The output sample is again a category.
@bm.random_variable
def classification(item: int, classifier: int):
    return Categorical(theta(classifier, category_of_item(item)))


class ClaraCategoricalTest(unittest.TestCase):
    def test_categorical_clara_inference(self) -> None:
        self.maxDiff = None

        # We observe the classifications our classifiers make of the
        # items:
        observations = {
            classification(foo_jpg, alice): tensor(sheep),
            classification(foo_jpg, bob): tensor(goat),
            classification(bar_jpg, alice): tensor(cat),
            classification(bar_jpg, bob): tensor(cat),
        }
        # We wish to know how likely it is that item is a sheep, goat or cat,
        # when given these observations.
        queries = [
            category_of_item(foo_jpg),
            category_of_item(bar_jpg),
        ]

        with self.assertRaises(ValueError) as ex:
            # TODO: Implement jitted stochastic control flows.
            # TODO: We should be able to accumulate the graph even if we cannot
            # make it work in BMG.
            BMGInference().to_dot(queries, observations, after_transform=False)

        observed = str(ex.exception)

        expected = "Jitted stochastic control flows are not yet implemented"
        self.assertEqual(expected.strip(), observed.strip())

        # TODO: Raise a better error than a generic ValueError
        with self.assertRaises(ValueError) as ex:
            BMGInference().infer(queries, observations, 10)
        observed = str(ex.exception)
        expected = "Jitted stochastic control flows are not yet implemented"
        self.assertEqual(expected.strip(), observed.strip())
