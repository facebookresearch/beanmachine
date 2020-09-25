# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torch
from beanmachine.ppl.experimental.neutra.iafmcmc_infer import IAFMCMCinference
from beanmachine.ppl.experimental.neutra.maskedautoencoder import MaskedAutoencoder
from beanmachine.ppl.testlib.abstract_conjugate import AbstractConjugateTests
from torch import nn


class SingleSiteIAFConjugateTest(unittest.TestCase, AbstractConjugateTests):
    # for some transformed spaces, the acceptance ratio is never .65
    # see Adapative HMC Conjugate Tests for more details
    def setUp(self):
        torch.manual_seed(17)

    def optimizer_func(self, lr, weight_decay):
        return lambda parameters: torch.optim.Adam(
            parameters, lr=1e-4, weight_decay=1e-5
        )

    def test_beta_binomial_conjugate_run(self):
        training_sample_size = 1
        length = 2
        in_layer = 1
        out_layer = 1
        hidden_layer = 30
        n_block = 4
        seed_num = 11
        network_architecture = MaskedAutoencoder(
            in_layer, out_layer, nn.ELU(), hidden_layer, n_block, seed_num
        )
        optimizer_func = self.optimizer_func(1e-4, 1e-5)

        iaf = IAFMCMCinference(
            training_sample_size,
            length,
            in_layer,
            network_architecture,
            optimizer_func,
            True,
            {},
        )
        self.beta_binomial_conjugate_run(
            iaf, num_samples=200, delta=0.3, num_adaptive_samples=100
        )

    def test_gamma_gamma_conjugate_run(self):
        training_sample_size = 10
        length = 2
        in_layer = 2
        out_layer = 4
        hidden_layer = 30
        n_block = 4
        seed_num = 11
        network_architecture = MaskedAutoencoder(
            in_layer, out_layer, nn.ELU(), hidden_layer, n_block, seed_num
        )

        optimizer_func = self.optimizer_func(1e-4, 1e-5)
        iaf = IAFMCMCinference(
            training_sample_size,
            length,
            in_layer,
            network_architecture,
            optimizer_func,
            True,
            [],
        )
        self.gamma_gamma_conjugate_run(
            iaf, num_samples=200, delta=0.3, num_adaptive_samples=100
        )

    def test_gamma_normal_conjugate_run(self):
        training_sample_size = 10
        length = 2
        in_layer = 2
        out_layer = 4
        hidden_layer = 30
        n_block = 4
        seed_num = 11
        network_architecture = MaskedAutoencoder(
            in_layer, out_layer, nn.ELU(), hidden_layer, n_block, seed_num
        )

        optimizer_func = self.optimizer_func(1e-4, 1e-5)
        iaf = IAFMCMCinference(
            training_sample_size,
            length,
            in_layer,
            network_architecture,
            optimizer_func,
            True,
            [],
        )
        self.gamma_normal_conjugate_run(
            iaf, num_samples=150, delta=0.3, num_adaptive_samples=50
        )

    def test_normal_normal_conjugate_run(self):
        training_sample_size = 10
        length = 2
        in_layer = 2
        out_layer = 4
        hidden_layer = 30
        n_block = 4
        seed_num = 11
        network_architecture = MaskedAutoencoder(
            in_layer, out_layer, nn.ELU(), hidden_layer, n_block, seed_num
        )

        optimizer_func = self.optimizer_func(1e-4, 1e-5)
        iaf = IAFMCMCinference(
            training_sample_size,
            length,
            in_layer,
            network_architecture,
            optimizer_func,
            True,
            [],
        )
        # TODO: The delta in the following needs to be reduced
        self.normal_normal_conjugate_run(
            iaf, num_samples=100, delta=0.33, num_adaptive_samples=500
        )

    def test_distant_normal_normal_conjugate_run(self):
        pass

    def test_dirichlet_categorical_conjugate_run(self):
        training_sample_size = 1
        length = 2
        in_layer = 1
        out_layer = 2
        hidden_layer = 30
        n_block = 4
        seed_num = 11

        network_architecture = MaskedAutoencoder(
            in_layer, out_layer, nn.ELU(), hidden_layer, n_block, seed_num
        )

        optimizer_func = self.optimizer_func(1e-4, 1e-5)
        iaf = IAFMCMCinference(
            training_sample_size,
            length,
            in_layer,
            network_architecture,
            optimizer_func,
            True,
            [],
        )
        # TODO: The delta in the following needs to be reduced
        self.dirichlet_categorical_conjugate_run(
            iaf, num_samples=200, delta=0.364, num_adaptive_samples=100
        )
