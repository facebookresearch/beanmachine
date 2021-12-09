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
from typing import List, Tuple

import torch
import torch.nn as nn
from beanmachine.ppl.experimental.neutra.iaflayer import InverseAutoregressiveLayer
from beanmachine.ppl.legacy.world import World
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from torch import Tensor


class InverseAutoregressiveFlow(nn.Module):
    """
    A IAF Model is a (prior, flow) pair. We try to calculate the evidence lower
    bound (ELBO) is calculated in z space instead of original θ space,using by
    default Eq (3) from Hoffman, Matthew, et al. (2019).
        L(φ) = A − KL(q(θ) || p(θ))
        = integrals q(z) log π(f(z))/(q(z)|∂f/∂z |^−1) dz  Eq (3)

    Thus we compute Monte Carlo estimate of the ELBO by evaluating the log-ratio
    in Eq (3) at a z sampled from q(z). Use estimates to maximize the ELBO in z
    space, and therefore minimize the KL divergence from q(θ) to p(θ).

    """

    def __init__(
        self,
        based_distribution,
        network_architecture,
        length: int,
        dim: int,
        node: RVIdentifier,
        world: World,
        stable: bool,
    ):

        """
        Define the inverse autoregressive layers and construct the inverse autoregressive
        flow based on several inverse autoregressive layers.

        :param base_distribution: the initialized distribution we start in z space
        to the approximate target distribution.
        :param world: the world in which we have already proposed a new value
        for node.
        :param stable: chose to use "stable" version of IAF or not.
        :param network_architecture: it is a parameter class that users give to
        define a masked_autoencoder network, includeing: input_layer, output_layer,
        activation function, hidden_layer, n_block, and seed number.
        :param length: how many IAF layers in the network.

        """

        super().__init__()
        self.based_distribution_ = based_distribution
        self.length_ = length
        self.world = world
        self.node = node

        flows_ = []
        # construct a inverse autoregressive flow
        for i in range(self.length_):
            flows_.append(
                InverseAutoregressiveLayer(network_architecture, i, stable, dim)
            )

        self.flows_ = nn.ModuleList(flows_)

    def forward(self, x: Tensor) -> Tuple[List[Tensor], Tensor, Tensor]:
        """
        the forward method that compute the f(z) = z*sigma+mu, log_jacobian
        and ELBO of IAF flow (has many IAF layers).

        :param x: the samples draw from previous distribution which later would
        be transofrmed by flows.
        :returns: f(z), ELBO, log_jacobian.
        """

        elbo = 0
        z_f = [x]
        len_j = len(x)
        log_jacobian = torch.zeros(len_j)
        for layer_ in self.flows_:
            # loop through the NN to add log_abs_jacobian to ELBO
            tmp = z_f[-1][:]
            x, log_d = layer_.forward(tmp)
            elbo += log_d
            z_f.append(x)
            if log_jacobian.size != log_d.size():
                log_jacobian = torch.zeros(log_d.size())
            log_jacobian += log_d
        elbo += (
            -self.based_distribution_.log_prob(z_f[0]).view(z_f[0].size(0), -1).sum(1)
        )

        self.world.propose_change_transformed_value(
            self.node, z_f[-1], allow_graph_update=False
        )

        node_var = self.world.get_node_in_world_raise_error(self.node, False)

        # compute_score would compute the log_prob of observation.
        score = self.world.compute_score(node_var)
        elbo += score
        self.world.reject_diff()
        return z_f, elbo, log_jacobian

    def backward(self, z: Tensor) -> Tuple[List[Tensor], Tensor]:
        """
        the backward method that compute the f(z) = z*sigma+mu,
        log_jacobian and ELBO in the inverse direction of the flow.

        :param z: the samples has been transformed by flows.
        :returns: f_z, which is the inverse map from f(z) to z,
        and log_jacobian. But we do not need to use the output in
        the training.

        """

        f_z = [z]
        len_j = len(z)
        log_jacobian = torch.zeros(len_j)
        for layer_ in self.flows_[::-1]:
            tmp = f_z[-1][:]
            z, log_d = layer_.backward(tmp)
            f_z.append(z)
            log_jacobian += log_d
        return f_z, log_jacobian
