# Copyright (c) Facebook, Inc. and its affiliates
from typing import Tuple

import torch
import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.inference.abstract_single_site_mh_infer import (
    AbstractSingleSiteMHInference,
)
from beanmachine.ppl.model.utils import RandomVariable
from beanmachine.ppl.world.variable import Variable
from torch import Tensor
from torch.autograd import grad


class SingleSiteNewtonianMonteCarlo(AbstractSingleSiteMHInference):
    """
    Single-Site Newtonian Monte Carlo Implementations

    In this implementation, we draw a new sample from a proposal that is a
    MultivariateNormal with followings specifications:
        mean = sampled_value - gradient * hessian_inversed
        covariance = - hessian_inversed

    Where the gradient and hessian are computed over the log probability of
    the node being resampled and log probabilities of its immediate children

    To compute, the proposal log update, we go through the following steps:
        1) Draw a new sample from MultivariateNormal(node.mean, node.covariance)
        and compute the log probability of the new draw coming from this
        proposal distribution log(P(X->X'))
        2) Construct the new diff given the new value
        3) Compute new gradient and hessian and hence, new mean and
        covariance and compute the log probability of the old value coming
        from MutlivariateNormal(new mean, new covariance) log(P(X'->X))
        4) Compute the final proposal log update: log(P(X'->X)) - log(P(X->X'))
    """

    def __init__(self):
        super().__init__()

    def compute_normal_mean_covar(self, node_var: Variable) -> Tuple[Tensor, Tensor]:
        """
        Computes mean and covariance of the MultivariateNormal given the node.

        :param node_var: the node Variable we're proposing a new value for
        :returns: mean and covariance
        """
        hessian = None
        node_val = node_var.value
        score = node_var.log_prob
        for child in node_var.children:
            score += self.world_.get_node_in_world(child, False).log_prob
        gradient = grad(score, node_val, create_graph=True)[0]
        first_gradient = gradient.reshape(-1).clone()
        size = first_gradient.shape[0]
        for i in range(size):
            second_gradient = (
                grad(
                    first_gradient.index_select(0, tensor([i])),
                    node_val,
                    create_graph=True,
                )[0]
            ).reshape(-1)
            hessian = (
                torch.cat((hessian, (second_gradient).unsqueeze(0)), 0)
                if hessian is not None
                else (second_gradient).unsqueeze(0)
            )

        if hessian is None:
            raise ValueError("Something went wrong with gradient computation")

        # to avoid problems iwth inverse, here we add a small value - 1e-7 to
        # the diagonals
        diag = (1e-7) * torch.eye(hessian.shape[0])
        hessian_inverse = (hessian + diag).inverse()
        # node value may of any arbitrary shape, so here, we transform it into a
        # 1D vector using reshape(-1) and with unsqueeze(0), we change 1D vector
        # of size (N) to (1 x N) matrix.
        node_reshaped = node_val.reshape(-1).unsqueeze(0)
        # here we again call unsqueeze(0) on first_gradient to transform it into
        # a matrix in order to be able to perform matrix multiplication.
        mean = (
            node_reshaped - first_gradient.unsqueeze(0).mm(hessian_inverse)
        ).squeeze(0)
        covariance = tensor(-1) * hessian_inverse
        return mean, covariance

    def post_process(self, node: RandomVariable) -> Tensor:
        """
        Computes new gradient and hessian and hence, new mean and covariance and
        finally computes the log probability of the old value coming from
        MutlivariateNormal(new mean, new covariance) log(P(X->X')).

        :param node: the node for which we have already proposed a new value for.
        :param proposal_log_update: proposal log update computed up until now
        which is log(P(X->X'))
        :returns: log(P(X'->X)) - log(P(X->X'))
        """
        node_var = self.world_.get_node_in_world(node, False)
        old_value = self.world_.variables_[node].value
        mean, covariance = self.compute_normal_mean_covar(node_var)
        new_value_dist = dist.MultivariateNormal(mean, covariance)
        node_var.mean = mean
        node_var.covariance = covariance
        return new_value_dist.log_prob(old_value)

    def propose(self, node: RandomVariable) -> Tuple[Tensor, Tensor]:
        """
        Proposes a new value for the node by drawing a sample from the proposal
        distribution and compute the log probability of the new draw coming from
        this proposal distribution log(P(X->X')).

        :param node: the node for which we'll need to propose a new value for.
        :returns: a new proposed value for the node and the difference in log_prob
        of the old and newly proposed value.
        """
        node_var = self.world_.get_node_in_world(node, False)
        node_val = node_var.value
        if node_var.mean is None and node_var.covariance is None:
            mean, covariance = self.compute_normal_mean_covar(node_var)
        else:
            mean = node_var.mean
            covariance = node_var.covariance

        new_value_dist = dist.MultivariateNormal(mean, covariance)
        new_value = new_value_dist.sample().reshape(node_val.shape)
        new_value.requires_grad_(True)
        negative_proposal_log_update = -1 * new_value_dist.log_prob(new_value)
        return (new_value, negative_proposal_log_update)
