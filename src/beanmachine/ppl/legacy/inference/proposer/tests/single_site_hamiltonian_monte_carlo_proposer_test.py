# Copyright (c) Facebook, Inc. and its affiliates
import unittest

import torch
from beanmachine.ppl.legacy.inference.proposer.single_site_hamiltonian_monte_carlo_proposer import (
    SingleSiteHamiltonianMonteCarloProposer,
)
from torch import tensor


class SingleSiteHamiltonianMonteCarloProposerTest(unittest.TestCase):
    def test_hmc_adapt_mass_matrix(self):
        proposer = SingleSiteHamiltonianMonteCarloProposer(0.5)
        sample = tensor(
            [
                [[1.0616, 0.6935, -0.8846], [0.3632, 0.3001, -1.0865]],
                [[-0.8563, 0.7314, 0.5012], [0.4715, 0.2294, -0.7616]],
                [[0.1780, 0.6197, 0.3229], [-0.1003, 0.3336, -0.7157]],
            ]
        )
        for i in range(3):
            proposer._adapt_mass_matrix(i + 1, sample[i])

        true_cov = tensor(
            [
                [0.9215, -0.0205, -0.6515, -0.0649, 0.0356, -0.1505],
                [-0.0205, 0.0032, -0.0028, 0.0170, -0.0028, -0.0034],
                [-0.6515, -0.0028, 0.5683, -0.0513, -0.0127, 0.1483],
                [-0.0649, 0.0170, -0.0513, 0.0922, -0.0138, -0.0271],
                [0.0356, -0.0028, -0.0127, -0.0138, 0.0028, -0.0010],
                [-0.1505, -0.0034, 0.1483, -0.0271, -0.0010, 0.0408],
            ]
        )
        delta = 1e-3
        padded_cov = delta * torch.eye(6) + (1 - delta) * true_cov
        for i in range(6):
            for j in range(6):
                self.assertAlmostEqual(
                    proposer.covariance[i, j].item(), padded_cov[i, j].item(), places=3
                )

    def test_hmc_adapt_mass_matrix_single_value(self):
        proposer = SingleSiteHamiltonianMonteCarloProposer(0.5)
        sample = tensor([1.0393, -0.5673, 1.0703, 1.2635, 0.8490])
        for i in range(len(sample)):
            proposer._adapt_mass_matrix(i + 1, sample[i])
        delta = 1e-3
        padded_cov = delta + (1 - delta) * sample.var().item()
        self.assertAlmostEqual(proposer.covariance.item(), padded_cov, places=6)

    def test_hmc_compute_kinetic_energy(self):
        proposer = SingleSiteHamiltonianMonteCarloProposer(0.5)
        proposer.covariance = tensor([[0.95, 0.05], [0.05, 0.60]])
        k = proposer._compute_kinetic_energy(tensor([1.0, 2.0]))
        self.assertAlmostEqual(k.item(), 1.775)
