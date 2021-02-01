#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# CLARA CGM model compiler test

import unittest
from typing import List

import beanmachine.ppl as bm
import torch
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch.distributions import Beta, Bernoulli


@bm.random_variable
def sensitivity(labeler):
    # TODO: Something is crashing in set_trace; a BdbQuit exception is
    # raised and it is not yet clear to me why. Commenting that out
    # until we understand why.

    # import pdb
    # pdb.set_trace()
    return Beta(8.0, 2.0)


@bm.random_variable
def specificity(labeler):
    return Beta(8.0, 2.0)


@bm.random_variable
def prevalence():
    return Beta(0.5, 0.5)


@bm.random_variable
def observation(dataset):
    log_prob = torch.tensor(0)
    for item in dataset.items:
        pos_sum = prevalence().log()
        neg_sum = (1 - prevalence()).log()

        for j in range(len(item.labels)):
            label = item.labels[j].rating.id
            labeler = item.labels[j].labeler.id
            # TODO: Note that += is still not working in the compiler
            if label == 1:
                pos_sum = pos_sum + sensitivity(labeler).log()
                neg_sum = neg_sum + (1 - specificity(labeler)).log()
            else:
                pos_sum = pos_sum + (1 - sensitivity(labeler)).log()
                neg_sum = neg_sum + specificity(labeler).log()

        log_prob_i = torch.tensor([pos_sum, neg_sum]).logsumexp(dim=0)
        log_prob = log_prob + log_prob_i
    return Bernoulli(log_prob.exp())


class Rating:
    id: int

    def __init__(self, id):
        self.id = id


cat = Rating(1)
dog = Rating(0)


class Labeler:
    id: int
    sensitivity: float
    specificity: float

    def __init__(self, sensitivity, specificity, id):
        self.sensitivity = sensitivity
        self.specificity = specificity
        self.id = id


bob = Labeler(0.90, 0.70, 0)
joe = Labeler(0.70, 0.80, 1)
sue = Labeler(0.90, 0.90, 2)


class Label:

    labeler: Labeler
    rating: Rating

    def __init__(self, labeler, rating):
        self.labeler = labeler
        self.rating = rating


bobcat = Label(bob, cat)
bobdog = Label(bob, dog)
joecat = Label(joe, cat)
joedog = Label(joe, dog)
suecat = Label(sue, cat)
suedog = Label(sue, dog)


class Item:
    true_rating: Rating
    labels: List[Label]
    id: str

    def __init__(self, true_rating, labels, id):
        self.true_rating = true_rating
        self.labels = labels
        self.id = id


class Dataset:
    items: List[Item]

    def __init__(self, items):
        self.items = items


dataset = Dataset(
    [
        Item(cat, [bobcat, joecat, suecat], "Batman"),
        Item(cat, [bobcat, joedog, suecat], "Pistachio"),
        Item(cat, [bobcat, joedog, suecat], "Loki"),
        Item(cat, [bobcat, joecat, suecat], "Tiger"),
        Item(cat, [bobcat, joecat, suecat], "Socks"),
        Item(cat, [bobdog, joecat, suecat], "Biscuit"),
        Item(cat, [bobcat, joecat, suecat], "Zorro"),
        Item(cat, [bobcat, joedog, suecat], "Asker"),
        Item(cat, [bobcat, joecat, suedog], "Smudge"),
        Item(cat, [bobcat, joecat, suecat], "Smoky"),
        Item(dog, [bobdog, joedog, suedog], "Mr. Piffles"),
        Item(dog, [bobdog, joecat, suedog], "Daisy"),
        Item(dog, [bobcat, joedog, suedog], "Cody"),
        Item(dog, [bobdog, joedog, suedog], "Rex"),
        Item(dog, [bobdog, joecat, suedog], "Fido"),
        Item(dog, [bobdog, joedog, suedog], "Bruce"),
        Item(dog, [bobcat, joedog, suedog], "Remy"),
        Item(dog, [bobcat, joedog, suedog], "Lassie"),
        Item(dog, [bobdog, joedog, suedog], "Pasta"),
        Item(dog, [bobdog, joedog, suecat], "Shep"),
    ]
)


class ClaraCGM2Test(unittest.TestCase):
    def test_clara_cgm_2_inference(self) -> None:
        self.maxDiff = None
        queries = [
            prevalence(),
            sensitivity(bob.id),
            specificity(bob.id),
            sensitivity(joe.id),
            specificity(joe.id),
            sensitivity(sue.id),
            specificity(sue.id),
        ]
        observations = {observation(dataset): torch.tensor(1.0)}
        num_samples = 1000
        mcsamples = BMGInference().infer(queries, observations, num_samples)
        prevalence_samples = mcsamples[prevalence()]
        observed = prevalence_samples.mean()
        expected = 0.5
        self.assertAlmostEqual(first=observed, second=expected, delta=0.05)

        sens_bob = mcsamples[sensitivity(bob.id)]
        observed = sens_bob.mean()
        expected = bob.sensitivity
        self.assertAlmostEqual(first=observed, second=expected, delta=0.15)

        sens_joe = mcsamples[sensitivity(joe.id)]
        observed = sens_joe.mean()
        expected = joe.sensitivity
        self.assertAlmostEqual(first=observed, second=expected, delta=0.15)

        sens_sue = mcsamples[sensitivity(sue.id)]
        observed = sens_sue.mean()
        expected = sue.sensitivity
        self.assertAlmostEqual(first=observed, second=expected, delta=0.15)

        spec_bob = mcsamples[specificity(bob.id)]
        observed = spec_bob.mean()
        expected = bob.specificity
        self.assertAlmostEqual(first=observed, second=expected, delta=0.15)

        spec_joe = mcsamples[specificity(joe.id)]
        observed = spec_joe.mean()
        expected = joe.specificity
        self.assertAlmostEqual(first=observed, second=expected, delta=0.15)

        spec_sue = mcsamples[specificity(sue.id)]
        observed = spec_sue.mean()
        expected = sue.specificity
        self.assertAlmostEqual(first=observed, second=expected, delta=0.15)
