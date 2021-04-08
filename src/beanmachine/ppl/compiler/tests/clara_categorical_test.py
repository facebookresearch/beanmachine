#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

# This is a simplified version of the CLARA model which uses categorical
# distributions. We do not support compiling this to BMG yet because
# we do not have categorical distributions in BMG or
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

        observed = BMGInference().to_dot(queries, observations, after_transform=False)
        expected = """
digraph "graph" {
  N00[label=0];
  N01[label="[10.0,5.0,1.0]"];
  N02[label=Dirichlet];
  N03[label=Sample];
  N04[label=1];
  N05[label="[5.0,10.0,1.0]"];
  N06[label=Dirichlet];
  N07[label=Sample];
  N08[label=2];
  N09[label="[1.0,1.0,10.0]"];
  N10[label=Dirichlet];
  N11[label=Sample];
  N12[label=map];
  N13[label="[1.0,1.0,1.0]"];
  N14[label=Dirichlet];
  N15[label=Sample];
  N16[label=Categorical];
  N17[label=Sample];
  N18[label=index];
  N19[label=Categorical];
  N20[label=Sample];
  N21[label="Observation tensor(0)"];
  N22[label=Sample];
  N23[label=Sample];
  N24[label=Sample];
  N25[label=map];
  N26[label=index];
  N27[label=Categorical];
  N28[label=Sample];
  N29[label="Observation tensor(1)"];
  N30[label=Sample];
  N31[label=index];
  N32[label=Categorical];
  N33[label=Sample];
  N34[label="Observation tensor(2)"];
  N35[label=index];
  N36[label=Categorical];
  N37[label=Sample];
  N38[label="Observation tensor(2)"];
  N39[label=Query];
  N40[label=Query];
  N00 -> N12;
  N00 -> N25;
  N01 -> N02;
  N02 -> N03;
  N02 -> N22;
  N03 -> N12;
  N04 -> N12;
  N04 -> N25;
  N05 -> N06;
  N06 -> N07;
  N06 -> N23;
  N07 -> N12;
  N08 -> N12;
  N08 -> N25;
  N09 -> N10;
  N10 -> N11;
  N10 -> N24;
  N11 -> N12;
  N12 -> N18;
  N12 -> N31;
  N13 -> N14;
  N14 -> N15;
  N15 -> N16;
  N16 -> N17;
  N16 -> N30;
  N17 -> N18;
  N17 -> N26;
  N17 -> N39;
  N18 -> N19;
  N19 -> N20;
  N20 -> N21;
  N22 -> N25;
  N23 -> N25;
  N24 -> N25;
  N25 -> N26;
  N25 -> N35;
  N26 -> N27;
  N27 -> N28;
  N28 -> N29;
  N30 -> N31;
  N30 -> N35;
  N30 -> N40;
  N31 -> N32;
  N32 -> N33;
  N33 -> N34;
  N35 -> N36;
  N36 -> N37;
  N37 -> N38;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # TODO: We do not yet support Categorical
        # or map/index nodes in BMG.  Revisit this test when we do.
        # TODO: Raise a better error than a generic ValueError
        # TODO: These error messages are needlessly repetitive.
        # Deduplicate them.
        # TODO: These error messages are phrased in terms of the
        # graph operations, not the source code.
        with self.assertRaises(ValueError) as ex:
            BMGInference().infer(queries, observations, 10)
        observed = str(ex.exception)
        expected = """
The model uses a Categorical operation unsupported by Bean Machine Graph.
The unsupported node is the operand of a Sample.
The model uses a Categorical operation unsupported by Bean Machine Graph.
The unsupported node is the operand of a Sample.
The model uses a Categorical operation unsupported by Bean Machine Graph.
The unsupported node is the operand of a Sample.
The model uses a Categorical operation unsupported by Bean Machine Graph.
The unsupported node is the operand of a Sample.
The model uses a Categorical operation unsupported by Bean Machine Graph.
The unsupported node is the operand of a Sample.
The model uses a index operation unsupported by Bean Machine Graph.
The unsupported node is the probability of a Categorical.
The model uses a index operation unsupported by Bean Machine Graph.
The unsupported node is the probability of a Categorical.
The model uses a index operation unsupported by Bean Machine Graph.
The unsupported node is the probability of a Categorical.
The model uses a index operation unsupported by Bean Machine Graph.
The unsupported node is the probability of a Categorical.
The model uses a map operation unsupported by Bean Machine Graph.
The unsupported node is the left of a index.
The model uses a map operation unsupported by Bean Machine Graph.
The unsupported node is the left of a index.
        """
        self.assertEqual(expected.strip(), observed.strip())
