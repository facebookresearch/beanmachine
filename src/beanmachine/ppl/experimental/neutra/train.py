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
import torch
import torch.distributions as dist
import torch.nn as nn
from beanmachine.ppl.experimental.neutra.iafflow import InverseAutoregressiveFlow
from beanmachine.ppl.legacy.world import World
from beanmachine.ppl.model.rv_identifier import RVIdentifier


class IAFMap(nn.Module):
    """
    Learn mapping f from IAF and get a series of mu and sigma.

    """

    def __init__(
        self,
        node: RVIdentifier,
        world: World,
        length: int,
        dim: int,
        network_architecture,
        stable: bool,
        num_sample: int,
    ):
        """
        Define the IAF model to train the mapping.

        :param node: data sampled from the based distribution
        :param world: the world in which we have already proposed a new value
        for the node.
        :param length: how many IAF layers are in the inverse autoregressive flow.
        :param dim: the dimension of the input layer.
        :param network_architecture: it is a parameter class to define a
        MaskedAutoencoder NN,
        including: input_layer, output_layer, activation function, hidden_layer,
        n_block, and seed number.
        :param stable: chose to use the "stable" version of IAF or not.
        :param num_sample: the number of samples draws from based
        distribution~N(0,1).

        """
        super().__init__()
        self.num_sample = num_sample
        based_distribution_ = dist.Normal(torch.zeros(1), torch.ones(1))
        self.model_ = InverseAutoregressiveFlow(
            based_distribution_, network_architecture, length, dim, node, world, stable
        )
        self.world = world
        self.node = node

    def train_iaf(self, world: World, optimizer) -> None:
        """
        train IAF model.
        :param world: the world in which we have already proposed a new value
        for the node.
        :param optimizer: the optimization algorithm you chose to optimize the
        parameters based on the computed gradients.
        :return: None

        """
        node_var = self.world.get_node_in_world_raise_error(self.node, False)
        shape_ = node_var.transformed_value.shape
        x = dist.Normal(torch.zeros(shape_), torch.ones(shape_)).sample(
            sample_shape=(self.num_sample,)
        )
        z_f, elbo, log_jacobian = self.model_(x)
        loss = -(torch.sum(elbo)) / self.num_sample
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
