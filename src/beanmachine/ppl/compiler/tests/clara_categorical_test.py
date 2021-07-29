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
  N00[label="[1.0,1.0,1.0]"];
  N01[label=Dirichlet];
  N02[label=Sample];
  N03[label=Categorical];
  N04[label=Sample];
  N05[label=0];
  N06[label="[10.0,5.0,1.0]"];
  N07[label=Dirichlet];
  N08[label=Sample];
  N09[label=1];
  N10[label="[5.0,10.0,1.0]"];
  N11[label=Dirichlet];
  N12[label=Sample];
  N13[label=2];
  N14[label="[1.0,1.0,10.0]"];
  N15[label=Dirichlet];
  N16[label=Sample];
  N17[label=Switch];
  N18[label=Categorical];
  N19[label=Sample];
  N20[label="Observation tensor(0)"];
  N21[label=Sample];
  N22[label=Sample];
  N23[label=Sample];
  N24[label=Switch];
  N25[label=Categorical];
  N26[label=Sample];
  N27[label="Observation tensor(1)"];
  N28[label=Sample];
  N29[label=Switch];
  N30[label=Categorical];
  N31[label=Sample];
  N32[label="Observation tensor(2)"];
  N33[label=Switch];
  N34[label=Categorical];
  N35[label=Sample];
  N36[label="Observation tensor(2)"];
  N37[label=Query];
  N38[label=Query];
  N00 -> N01;
  N01 -> N02;
  N02 -> N03;
  N03 -> N04;
  N03 -> N28;
  N04 -> N17;
  N04 -> N24;
  N04 -> N37;
  N05 -> N17;
  N05 -> N24;
  N05 -> N29;
  N05 -> N33;
  N06 -> N07;
  N07 -> N08;
  N07 -> N21;
  N08 -> N17;
  N08 -> N29;
  N09 -> N17;
  N09 -> N24;
  N09 -> N29;
  N09 -> N33;
  N10 -> N11;
  N11 -> N12;
  N11 -> N22;
  N12 -> N17;
  N12 -> N29;
  N13 -> N17;
  N13 -> N24;
  N13 -> N29;
  N13 -> N33;
  N14 -> N15;
  N15 -> N16;
  N15 -> N23;
  N16 -> N17;
  N16 -> N29;
  N17 -> N18;
  N18 -> N19;
  N19 -> N20;
  N21 -> N24;
  N21 -> N33;
  N22 -> N24;
  N22 -> N33;
  N23 -> N24;
  N23 -> N33;
  N24 -> N25;
  N25 -> N26;
  N26 -> N27;
  N28 -> N29;
  N28 -> N33;
  N28 -> N38;
  N29 -> N30;
  N30 -> N31;
  N31 -> N32;
  N33 -> N34;
  N34 -> N35;
  N35 -> N36;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # TODO: The obscure "switch" error here is a result of an unsupported
        # stochastic control flow for which we do not yet produce a useful
        # error message.
        #
        # The unsupported control flow is theta(classifier, category_of_item(item));
        # category_of_item is a small natural, but theta produces a simplex, and
        # we have no way yet to build a graph to represent the operation "choose
        # one of these simplexes from a list of simplexes".
        #
        # That is, we have k samples from theta, each of which is of type simplex, and
        # we have category_of_item, which is a number from 0 to k-1.  But we cannot
        # use TO_MATRIX on those k samples because TO_MATRIX requires its inputs to be
        # all atomic values; it does not paste together a bunch of column simplexes into
        # a column simplex matrix from which we can select a single column via indexing.
        #
        # What we need to do is either (1) make TO_MATRIX do that, (2) make a new operation
        # similar to TO_MATRIX for gluing together columns, or (3) make a generalized
        # IF_THEN_ELSE that chooses from k choices rather than two choices.  (And of course
        # it is possible to do more than one of these options; they are more generally useful
        # than just solving this problem alone.)

        # TODO: Raise a better error than a generic ValueError
        # TODO: These error messages are needlessly repetitive.
        # Deduplicate them.
        # TODO: These error messages are phrased in terms of the
        # graph operations, not the source code.
        with self.assertRaises(ValueError) as ex:
            BMGInference().infer(queries, observations, 10)
        observed = str(ex.exception)
        expected = """
The model uses a Switch operation unsupported by Bean Machine Graph.
The unsupported node is the probability of a Categorical.
The model uses a Switch operation unsupported by Bean Machine Graph.
The unsupported node is the probability of a Categorical.
The model uses a Switch operation unsupported by Bean Machine Graph.
The unsupported node is the probability of a Categorical.
The model uses a Switch operation unsupported by Bean Machine Graph.
The unsupported node is the probability of a Categorical.
        """
        self.assertEqual(expected.strip(), observed.strip())
