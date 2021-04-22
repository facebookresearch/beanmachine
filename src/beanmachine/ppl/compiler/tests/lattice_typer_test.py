# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl.compiler.bmg_types as bt
import torch
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.lattice_typer import LatticeTyper


class LatticeTyperTest(unittest.TestCase):
    def test_lattice_typer_1(self) -> None:
        self.maxDiff = None
        bmg = BMGraphBuilder()
        typer = LatticeTyper()

        # Lattice type of a constant is based on its value.
        c0 = bmg.add_constant(0.0)
        self.assertEqual(bt.Zero, typer[c0])
        c1 = bmg.add_constant(1.0)
        self.assertEqual(bt.One, typer[c1])
        c2 = bmg.add_constant(2.0)
        self.assertEqual(bt.Natural, typer[c2])
        c3 = bmg.add_constant(1.5)
        self.assertEqual(bt.PositiveReal, typer[c3])
        c4 = bmg.add_constant(-1.5)
        self.assertEqual(bt.NegativeReal, typer[c4])
        c5 = bmg.add_constant(0.5)
        self.assertEqual(bt.Probability, typer[c5])
        c6 = bmg.add_constant(torch.tensor([1.5, -1.5]))
        self.assertEqual(bt.Real.with_dimensions(1, 2), typer[c6])

        # Lattice type of distributions is fixed:
        d0 = bmg.add_beta(c2, c2)
        prob = bmg.add_sample(d0)
        self.assertEqual(bt.Probability, typer[prob])

        d1 = bmg.add_bernoulli(prob)
        bo = bmg.add_sample(d1)
        self.assertEqual(bt.Boolean, typer[bo])

        d2 = bmg.add_binomial(c2, prob)
        nat = bmg.add_sample(d2)
        self.assertEqual(bt.Natural, typer[nat])

        d3 = bmg.add_halfcauchy(c3)
        posr = bmg.add_sample(d3)
        self.assertEqual(bt.PositiveReal, typer[posr])

        negr = bmg.add_negate(posr)
        self.assertEqual(bt.NegativeReal, typer[negr])

        d4 = bmg.add_normal(c0, c1)
        re = bmg.add_sample(d4)
        self.assertEqual(bt.Real, typer[re])

        # Lattice type of unsupported distributions and all descendents
        # is "untypable".

        d5 = bmg.add_chi2(c2)
        unt1 = bmg.add_sample(d5)
        unt2 = bmg.add_addition(unt1, unt1)
        self.assertEqual(bt.Untypable, typer[unt1])
        self.assertEqual(bt.Untypable, typer[unt2])

        # Spot check some operators.

        add1 = bmg.add_addition(prob, nat)
        self.assertEqual(bt.PositiveReal, typer[add1])

        pow1 = bmg.add_power(prob, posr)
        self.assertEqual(bt.Probability, typer[pow1])

        # TODO: Add more operators
