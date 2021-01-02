#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# CLARA CGM model compiler test

import unittest
from typing import Dict

import beanmachine.ppl as bm
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import tensor
from torch.distributions import Bernoulli, Beta


# The idea here is:
#
# * We have some set of "items" to be classified.
# * Each item belongs to exactly one of two "classes" which we represent
#   as zero (negative) or one (positive) -- but we do not know which.
# * The *overall* probability that a random item is positive is
#   "prevalence". We do not know the exact prevalence.
# * For each item some number of "labelers" produces a label judging
#   the item to be negative or positive.
# * The probability that a labeler correctly identifies a positive item
#   is their "sensitivity".
# * The probability that a labeler correctly identifies a negative item
#   is their "specificity".

# This is some constant between 0.0 and 1.0 that we choose.
expected_correctness = 0.8

# This is some positive constant that we choose; the higher the concentration,
# the closer we expect the sensitivity and specificity to be to the expected
# correctness.
concentration = 10

# we model each labeler's sensitivity and specificity as a beta distribution
# based on the expected correctness (ec) and concentraction (c):
#
# sensitivity_l ~ Beta(ec * c, (1 - ec) * c)
# specificity_l ~ Beta(ec * c, (1 - ec) * c)
# prevalence ~ Beta(0.5, 0.5)
#
# Our prior on prevalence is equal probability of both classes, but
# that the true prevalence is slightly more likely to be close to 0.0 or 1.0
# than close to the middle.
#
# We then compute:
#
# log_prob = 0
# for each item i
#    log_prob += log P(labels of item i | sensitivity, specificity, prevalence)

# Let's generate some sample data to test the model.

# Labels: is this a picture of a cat (positive) or dog (negative)?

dog = 0
cat = 1

# Labelers: each labeler has a true sensitivity and specificity.
class Labeler:
    sensitivity: float
    specificity: float
    name: str

    def __init__(self, sensitivity, specificity, name):
        self.sensitivity = sensitivity
        self.specificity = specificity
        self.name = name


# Given our prior of Beta(8, 2) for the sensitivity and specificity,
# we'd expect all of them to be around 0.60 to 0.95 or thereabouts.

bob = Labeler(0.90, 0.70, "Bob")
# When Bob gets a cat, he applies the cat label 90% of the time.
# When Bob gets a dog, he applies the dog label 70% of the time.

joe = Labeler(0.70, 0.80, "Joe")
sue = Labeler(0.90, 0.90, "Sue")

# Every item has the label assignments of each labeler. Note that not
# every labeler necessarily labels each item, though that is true
# in our sample data.
class Item:
    true_label: int
    labels: Dict[Labeler, int]
    name: str

    def __init__(self, true_label: int, labels: Dict[Labeler, int], name: str):
        self.true_label = true_label
        self.labels = labels
        self.name = name


# Let's suppose we have as many cats as dogs, and every labeler makes
# a judgment as to its class:

items = [
    Item(cat, {bob: cat, joe: cat, sue: cat}, "Batman"),
    Item(cat, {bob: cat, joe: dog, sue: cat}, "Pistachio"),
    Item(cat, {bob: cat, joe: dog, sue: cat}, "Loki"),
    Item(cat, {bob: cat, joe: cat, sue: cat}, "Tiger"),
    Item(cat, {bob: cat, joe: cat, sue: cat}, "Socks"),
    Item(cat, {bob: dog, joe: cat, sue: cat}, "Biscuit"),
    Item(cat, {bob: cat, joe: cat, sue: cat}, "Zorro"),
    Item(cat, {bob: cat, joe: dog, sue: cat}, "Asker"),
    Item(cat, {bob: cat, joe: cat, sue: dog}, "Smudge"),
    Item(cat, {bob: cat, joe: cat, sue: cat}, "Smoky"),
    Item(dog, {bob: dog, joe: dog, sue: dog}, "Mr. Piffles"),
    Item(dog, {bob: dog, joe: cat, sue: dog}, "Daisy"),
    Item(dog, {bob: cat, joe: dog, sue: dog}, "Cody"),
    Item(dog, {bob: dog, joe: dog, sue: dog}, "Rex"),
    Item(dog, {bob: dog, joe: cat, sue: dog}, "Fido"),
    Item(dog, {bob: dog, joe: dog, sue: dog}, "Bruce"),
    Item(dog, {bob: cat, joe: dog, sue: dog}, "Remy"),
    Item(dog, {bob: cat, joe: dog, sue: dog}, "Lassie"),
    Item(dog, {bob: dog, joe: dog, sue: dog}, "Pasta"),
    Item(dog, {bob: dog, joe: dog, sue: cat}, "Shep"),
]


@bm.random_variable
def sensitivity(labeler):
    return Beta(
        expected_correctness * concentration, (1 - expected_correctness) * concentration
    )


@bm.random_variable
def specificity(labeler):
    return Beta(
        expected_correctness * concentration, (1 - expected_correctness) * concentration
    )


@bm.random_variable
def prevalence():
    return Beta(0.5, 0.5)


# here, we compute P(labels | sens, spec, prev) for all items
# For a given item, P(labels for item | sens, spec, prev)
# can be broken down into
# P(labels for item | sens, spec, prev, true_label = dog) * P(true_label = dog) +
# P(labels for item | sens, spec, prev, true_label = cat) * P(true_label = cat)
@bm.random_variable
def observation():
    log_prob = 0
    for item in items:
        pos_sum = prevalence().log()
        # pos_sum holds log(P(labels | true_label = cat) * P(true_label = cat))
        neg_sum = (1 - prevalence()).log()
        # neg_sum holds log(P(labels | true_label = dog) * P(true_label = dog))
        for (labeler, label) in enumerate(item.labels):
            if label == cat:
                # TODO: Implement += in the compiler
                pos_sum = pos_sum + sensitivity(labeler).log()
                neg_sum = neg_sum + (1 - specificity(labeler)).log()
            else:
                pos_sum = pos_sum + (1 - sensitivity(labeler)).log()
                neg_sum = neg_sum + specificity(labeler).log()

        # TODO: Implement
        # TODO: torch.logsumexp(torch.tensor([pos_sum, neg_sum]), dim=0)
        # TODO: in the compiler

        log_prob_item = (pos_sum.exp() + neg_sum.exp()).log()
        log_prob = log_prob + log_prob_item
    return Bernoulli(log_prob.exp())


class ClaraCGMTest(unittest.TestCase):
    def test_clara_cgm_inference(self) -> None:
        self.maxDiff = None

        queries = [
            prevalence(),
            sensitivity(bob),
            sensitivity(joe),
            sensitivity(sue),
            specificity(bob),
            specificity(joe),
            specificity(sue),
        ]
        observations = {observation(): tensor(1.0)}

        num_samples = 1000
        inference = BMGInference()
        mcsamples = inference.infer(queries, observations, num_samples)
        prevalence_samples = mcsamples[prevalence()]
        observed = prevalence_samples.mean()
        # TODO: What is going on here? Should be around 0.5.
        expected = 0.02
        self.assertAlmostEqual(first=observed, second=expected, delta=0.05)

        sens_bob = mcsamples[sensitivity(bob)]
        observed = sens_bob.mean()
        expected = bob.sensitivity
        self.assertAlmostEqual(first=observed, second=expected, delta=0.15)

        sens_joe = mcsamples[sensitivity(joe)]
        observed = sens_joe.mean()
        expected = joe.sensitivity
        self.assertAlmostEqual(first=observed, second=expected, delta=0.15)

        sens_sue = mcsamples[sensitivity(sue)]
        observed = sens_sue.mean()
        expected = sue.sensitivity
        self.assertAlmostEqual(first=observed, second=expected, delta=0.15)

        spec_bob = mcsamples[specificity(bob)]
        observed = spec_bob.mean()
        expected = bob.specificity
        self.assertAlmostEqual(first=observed, second=expected, delta=0.15)

        spec_joe = mcsamples[specificity(joe)]
        observed = spec_joe.mean()
        expected = joe.specificity
        self.assertAlmostEqual(first=observed, second=expected, delta=0.15)

        spec_sue = mcsamples[specificity(sue)]
        observed = spec_sue.mean()
        expected = sue.specificity
        self.assertAlmostEqual(first=observed, second=expected, delta=0.15)
