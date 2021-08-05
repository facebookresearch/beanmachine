#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

# This is a simplified version of the CLARA model which uses categorical
# distributions.
#
# This module tests whether we can produce a legal BMG graph for this model.
# TODO: Actual inference is not yet implemented because (1) we do not have
# NMC inference implemented in BMG for graphs which contain natural values,
# and (2) categorical distribution does not yet compute gradients correctly.

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

        observed = BMGInference().to_dot(queries, observations)
        expected = """
digraph "graph" {
  N00[label="[1.0,1.0,1.0]"];
  N01[label=Dirichlet];
  N02[label=Sample];
  N03[label=Categorical];
  N04[label=Sample];
  N05[label="[10.0,5.0,1.0]"];
  N06[label=Dirichlet];
  N07[label=Sample];
  N08[label="[5.0,10.0,1.0]"];
  N09[label=Dirichlet];
  N10[label=Sample];
  N11[label="[1.0,1.0,10.0]"];
  N12[label=Dirichlet];
  N13[label=Sample];
  N14[label=Choice];
  N15[label=Categorical];
  N16[label=Sample];
  N17[label="Observation 0"];
  N18[label=Sample];
  N19[label=Sample];
  N20[label=Sample];
  N21[label=Choice];
  N22[label=Categorical];
  N23[label=Sample];
  N24[label="Observation 1"];
  N25[label=Sample];
  N26[label=Choice];
  N27[label=Categorical];
  N28[label=Sample];
  N29[label="Observation 2"];
  N30[label=Choice];
  N31[label=Categorical];
  N32[label=Sample];
  N33[label="Observation 2"];
  N34[label=Query];
  N35[label=Query];
  N00 -> N01;
  N01 -> N02;
  N02 -> N03;
  N03 -> N04;
  N03 -> N25;
  N04 -> N14;
  N04 -> N21;
  N04 -> N34;
  N05 -> N06;
  N06 -> N07;
  N06 -> N18;
  N07 -> N14;
  N07 -> N26;
  N08 -> N09;
  N09 -> N10;
  N09 -> N19;
  N10 -> N14;
  N10 -> N26;
  N11 -> N12;
  N12 -> N13;
  N12 -> N20;
  N13 -> N14;
  N13 -> N26;
  N14 -> N15;
  N15 -> N16;
  N16 -> N17;
  N18 -> N21;
  N18 -> N30;
  N19 -> N21;
  N19 -> N30;
  N20 -> N21;
  N20 -> N30;
  N21 -> N22;
  N22 -> N23;
  N23 -> N24;
  N25 -> N26;
  N25 -> N30;
  N25 -> N35;
  N26 -> N27;
  N27 -> N28;
  N28 -> N29;
  N30 -> N31;
  N31 -> N32;
  N32 -> N33;
}"""
        self.assertEqual(expected.strip(), observed.strip())

        # TODO: When inference is supported on categoricals, test:
        # BMGInference().infer(queries, observations, 10)
