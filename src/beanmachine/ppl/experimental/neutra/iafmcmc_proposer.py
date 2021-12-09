# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""

Implements inverse autoregressive flows.

reference:

Germain, Mathieu, et al. "Made: Masked autoencoder for distribution estimation."
International Conference on Machine Learning. 2015.
http://proceedings.mlr.press/v37/germain15.pdf
(MADE)

Improved Variational Inference with Inverse Autoregressive Flow, Kingma et al June 2016
https://arxiv.org/abs/1606.04934
(IAF)

MIT License, this work refers to an open source from Github, you can find the original code here:
https://github.com/karpathy/pytorch-normalizing-flows/blob/b60e119b37be10ce2930ef9fa17e58686aaf2b3d/nflib/made.py#L1
https://github.com/karpathy/pytorch-normalizing-flows/blob/b60e119b37be10ce2930ef9fa17e58686aaf2b3d/nflib/flows.py#L169


"""
from typing import Dict, Tuple

import torch
import torch.distributions as dist
from beanmachine.ppl.experimental.neutra.train import IAFMap
from beanmachine.ppl.legacy.inference.proposer.abstract_single_site_proposer import (
    AbstractSingleSiteProposer,
)
from beanmachine.ppl.legacy.world import World
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from torch import Tensor, tensor


class IAFMCMCProposer(AbstractSingleSiteProposer):
    """
    For random variables in theta space, returns a sample with
    (a transformation f added to theta, p(z)=p(theta=f(z)|df/dz|)
    added to the current variable.

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
    ):

        """
        Define the IAFMCMC Proposer.

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

        """

        self.training_sample_size = training_sample_size
        self.length = length
        self.in_layer = in_layer
        self.network_architecture = network_architecture
        self.stable = stable
        self.proposer_ = proposer
        self.mapping = None
        self.model = None
        self.optimizer = None
        self.sample_list_ = []
        self.optimizer_func = optimizer_func

        super().__init__()

    def do_adaptation(
        self,
        node: RVIdentifier,
        world: World,
        acceptance_probability: Tensor,
        num_adaptive_samples: int,
        is_accepted: bool,
        iteration_number: int,
    ) -> None:
        """
        adapted from learning mu and sigma of f mapping.

        :param node: the node for which we have already proposed a new value for.
        :param world: the world in which we have already proposed a new value
        for node.
        :param acceptance_probability: the acceptance probability of the previous move.
        :param iteration_number: The current iteration of inference
        :param num_adapt_steps: The number of inference iterations for adaptation.
        :param is_accepted: bool representing whether the new value was accepted.
        :returns: Nothing.

        """
        if iteration_number == 1:
            self.model = IAFMap(
                node,
                world,
                self.length,
                self.in_layer,
                self.network_architecture,
                self.stable,
                self.training_sample_size,
            )
            self.optimizer = self.optimizer_func(self.model.model_.parameters())
        self.model.train_iaf(world, self.optimizer)

        if iteration_number == num_adaptive_samples - 1:
            self.mapping = self.model.model_.eval()
            return

    def propose(self, node: RVIdentifier, world: World) -> Tuple[Tensor, Tensor, Dict]:
        """
        :param node: the node for which we'll need to propose a new value for.
        :param world: the world in which we'll propose a new value for node.
        :returns: a new proposed value for the node and the -ve log probability of
        proposing this new value and auxiliary variables that needs to be passed
        to post process.
        """
        node_var = world.get_node_in_world_raise_error(node, False)
        shape_ = node_var.transformed_value.shape
        x = dist.Normal(torch.zeros(shape_), torch.ones(shape_)).sample()
        x = x.unsqueeze(0)

        f_z, elbo, ja = self.mapping(x)
        self.sample_list_.append((f_z[-1], x, ja))
        proposed_value = node_var.inverse_transform_value(f_z[-1])
        proposed_value = proposed_value.squeeze()
        proposed_log_prob = (
            dist.Normal(0.0, 1.0).log_prob(x).sum() - ja.sum() + node_var.jacobian
        )
        return proposed_value, -proposed_log_prob, {}

    def post_process(
        self, node: RVIdentifier, world: World, auxiliary_variables: Dict
    ) -> Tensor:
        """
        :param node: the node for which we have already proposed a new value for.
        :param world: the world in which we have already proposed a new value
        for node.
        :param auxiliary_variables: Dict of auxiliary variables that is passed
        from propose.
        :returns: the log probability of proposing the old value from this new world.
        """
        node_var = world.get_node_in_world_raise_error(node, False)
        old_node_var = world.get_node_earlier_version(node)
        old_value = world.get_old_value(node)
        if old_node_var is not None and old_value is not None:
            old_transform_value = old_node_var.transform_value(old_value)
            if_find = False
            for sample_ in self.sample_list_:
                if torch.allclose(sample_[0], old_transform_value):
                    _, x_, ja = sample_
                    if_find = True
                    break
            if not if_find:
                return tensor(0.0)
        else:
            return tensor(0.0)

        old_proposed_log_prob = (
            dist.Normal(0.0, 1.0).log_prob(x_).sum() - ja.sum() + node_var.jacobian
        )
        return old_proposed_log_prob
