# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

from beanmachine.ppl.experimental.neutra.iafmcmc_proposer import IAFMCMCProposer
from beanmachine.ppl.legacy.inference.abstract_mh_infer import AbstractMHInference
from beanmachine.ppl.legacy.world import TransformType
from beanmachine.ppl.model.rv_identifier import RVIdentifier


class IAFMCMCinference(AbstractMHInference):
    """

    Implementation for Inverse Autoregressive Flow to transform the space of
    unfavorable geometries to isotropic Normal space, and run MCMC inference
    methods on the transformed space.

    """

    def __init__(
        self,
        training_sample_size: int,
        length: int,
        in_layer: int,
        network_architecture,
        optimizer_func,
        stable: bool,
        proposer,
        transform_type: TransformType = TransformType.DEFAULT,
        transforms: Optional[List] = None,
        skip_single_inference_run: bool = True,
    ):

        """
        Define the IAFMCMC inference.

        :param training_sample_size: the number of training samples draws from
        based distribution~N(0,1) to train the IAF network.
        :param length: how many IAF layers are in the inverse autoregressive flow.
        :param in_layer: the dimension of the input layer.
        :param network_architecture: it is a parameter class to define a
        MaskedAutoencoder NN, including: input_layer, output_layer, activation
        function, hidden_layer, n_block, and seed number.
        :param optimizer_func: callable returning a torch.optim to optimize
        model parameters with
        :param stable: chose to use the "stable" version of IAF or not.
        :param proposer: the default proposer.
        :param transform_type:
        :param transforms:
        :param skip_single_inference_run: skip calling the single_inference_run during
        the do adaptation step in IAF proposer.

        """

        super().__init__(
            proposer=IAFMCMCProposer(
                training_sample_size,
                length,
                in_layer,
                network_architecture,
                optimizer_func,
                stable,
                proposer,
            ),
            transform_type=transform_type,
            transforms=transforms,
        )
        self.skip_single_inference_run = skip_single_inference_run
        self.training_sample_size = training_sample_size
        self.network_architecture = network_architecture
        self.length = length
        self.in_layer = in_layer
        self.network_architecture = network_architecture
        self.optimizer_func = optimizer_func
        self.stable = stable
        self.proposer_ = {}
        self.proposer = []

    def find_best_single_site_proposer(self, node: RVIdentifier):
        """
        Finds the best proposer for a node.

        :param node: the node for which to return a proposer
        :returns: a proposer for the node
        """

        if node not in self.proposer_:
            self.proposer_[node] = IAFMCMCProposer(
                self.training_sample_size,
                self.length,
                self.in_layer,
                self.network_architecture,
                self.optimizer_func,
                self.stable,
                self.proposer,
            )
        return self.proposer_[node]
