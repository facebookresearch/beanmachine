# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl.compiler.bmg_types as bt
import torch
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.lattice_typer import LatticeTyper
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from torch import Size


def _rv_id() -> RVIdentifier:
    return RVIdentifier(lambda a, b: a, (1, 1))


class LatticeTyperTest(unittest.TestCase):
    def test_lattice_typer_matrix_ops(self) -> None:
        self.maxDiff = None
        bmg = BMGraphBuilder()
        typer = LatticeTyper()

        # create non constant real matrix
        zeros = bmg.add_real_matrix(torch.zeros(2, 2))
        ones = bmg.add_pos_real_matrix(torch.ones(2, 2))
        tensor_elements = []
        for row in range(0, 2):
            row_node = bmg.add_natural(row)
            row_mu = bmg.add_column_index(zeros, row_node)
            row_sigma = bmg.add_column_index(ones, row_node)
            for column in range(0, 2):
                index_node = bmg.add_natural(column)
                index_mu = bmg.add_vector_index(row_mu, index_node)
                index_sigma = bmg.add_vector_index(row_sigma, index_node)
                normal = bmg.add_normal(index_mu, index_sigma)
                sample = bmg.add_sample(normal)
                tensor_elements.append(sample)
        real_matrix = bmg.add_tensor(Size([2, 2]), *tensor_elements)

        # create non constant bool matrix
        probs = bmg.add_real_matrix(torch.tensor([[0.75, 0.25], [0.125, 0.875]]))
        tensor_elements = []
        for row in range(0, 2):
            row_node = bmg.add_natural(row)
            row_prob = bmg.add_column_index(probs, row_node)
            for column in range(0, 2):
                col_index = bmg.add_natural(column)
                prob = bmg.add_vector_index(row_prob, col_index)
                bernoulli = bmg.add_bernoulli(prob)
                sample = bmg.add_sample(bernoulli)
                tensor_elements.append(sample)
        bool_matrix = bmg.add_tensor(Size([2, 2]), *tensor_elements)

        neg_real = bmg.add_neg_real_matrix(torch.tensor([[-1.2, -1.3], [-4.7, -1.2]]))
        pos_real = bmg.add_matrix_exp(real_matrix)

        add_pos_to_reg = bmg.add_matrix_addition(pos_real, neg_real)
        mult_pos_to_neg = bmg.add_elementwise_multiplication(pos_real, neg_real)
        sum_bool = bmg.add_matrix_sum(bool_matrix)
        bmg.add_query(sum_bool, _rv_id())

        tpe_neg_real = typer[neg_real]
        tpe_real = typer[real_matrix]
        tpe_pos_real = typer[pos_real]

        tpe_add = typer[add_pos_to_reg]
        tpe_mult = typer[mult_pos_to_neg]
        tpe_sum = typer[sum_bool]

        self.assertTrue(isinstance(tpe_real, bt.RealMatrix))
        self.assertTrue(isinstance(tpe_neg_real, bt.NegativeRealMatrix))
        self.assertTrue(isinstance(tpe_pos_real, bt.PositiveRealMatrix))
        self.assertTrue(isinstance(tpe_add, bt.RealMatrix))
        self.assertTrue(isinstance(tpe_mult, bt.RealMatrix))
        self.assertTrue(isinstance(tpe_sum, bt.BooleanMatrix))

    def test_lattice_typer_1(self) -> None:
        self.maxDiff = None
        bmg = BMGraphBuilder()
        typer = LatticeTyper()

        # Lattice type of an untyped constant is based on its value.
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

        # BMG type of tensor is given assuming that when we emit it into
        # the BMG graph, it will be transposed into column-major form.
        # In BMG, it will be [[1.5], [-1.5]] and therefore this tensor is
        # typed as having two rows, one column, not one row, two columns
        # as it does in torch.
        c6 = bmg.add_constant(torch.tensor([1.5, -1.5]))
        self.assertEqual(bt.Real.with_dimensions(2, 1), typer[c6])

        # Lattice type of a typed constant is based on its type,
        # not its value. This real node is a real, even though its
        # value fits into a natural.
        c7 = bmg.add_real(2.0)
        self.assertEqual(bt.Real, typer[c7])

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
