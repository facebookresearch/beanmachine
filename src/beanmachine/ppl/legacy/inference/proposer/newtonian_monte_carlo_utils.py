# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Union, Callable

import torch
import torch.distributions as dist
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.utils import tensorops
from beanmachine.ppl.world import World
from torch import Tensor, tensor
from torch.autograd import grad


def is_scalar(val: Union[float, Tensor]) -> bool:
    """
    :returns: whether val is a scalar
    """
    return isinstance(val, float) or (isinstance(val, Tensor) and not val.shape)


def is_valid(vec: Tensor) -> bool:
    """
    :returns: whether a tensor is valid or not (not nan and not inf)
    """
    return not (torch.isnan(vec).any() or torch.isinf(vec).any())


def zero_grad(node_val: Tensor) -> None:
    """
    Zeros the gradient.
    """
    # pyre-fixme
    if node_val.is_leaf and hasattr(node_val, "grad") and node_val.grad is not None:
        node_val.grad.zero_()


def compute_first_gradient(
    score: Tensor,
    node_val: Tensor,
    create_graph: bool = False,
    retain_graph: Optional[bool] = None,
) -> Tuple[bool, Tensor]:
    """
    Computes the first gradient.

    :param score: the score to compute the gradient of
    :param node_val: the value to compute the gradient against
    :returns: the first gradient
    """
    if not node_val.requires_grad:
        raise ValueError("requires_grad_ needs to be set for node values")

    # pyre expects attributes to be defined in constructors or at class
    # top levels and doesn't support attributes that get dynamically added.
    # pyre-fixme
    elif node_val.is_leaf and node_val.grad is not None:
        node_val.grad.zero_()

    first_gradient = grad(
        score, node_val, create_graph=create_graph, retain_graph=retain_graph
    )[0]
    return is_valid(first_gradient), first_gradient


def compute_hessian(first_gradient: Tensor, node_val: Tensor) -> Tuple[bool, Tensor]:
    """
    Computes the hessian

    :param first_gradient: the first gradient of score with respect to
    node_val
    :param node_val: the value to compute the hessian against
    :returns: computes hessian
    """
    hessian = None
    size = first_gradient.shape[0]
    for i in range(size):
        second_gradient = (
            grad(
                # pyre-fixme
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

    if not is_valid(hessian):
        return False, tensor(0.0)

    return True, hessian


def soft_abs_inverse(neg_hessian: Tensor, alpha: float = 1e6) -> Tuple[Tensor, Tensor]:
    """
    Compute inverse of a symmetric matrix and returns inverse, eigen values
    and eigen vectors.

    :param neg_hessian: the value that we'd like to compute the inverse of
    :param alpha: the hardness parameter alpha for the SoftAbs map, see
    (https://arxiv.org/pdf/1212.4693.pdf)
    :returns: eigen value and eigen vector of the negative hessian inverse
    """
    eig_vals, eig_vecs = torch.linalg.eigh(neg_hessian)
    inverse_eig_vals = torch.tanh(alpha * eig_vals) / eig_vals
    return eig_vecs, inverse_eig_vals


def compute_eigvals_eigvecs(
    score: Tensor, node_val: Tensor
) -> Tuple[bool, Tensor, Tensor, Tensor]:
    """
    Compute hessian and returns eigen values and eigen vectors of the negative
    hessian inverse.

    :param score: the score function
    :param node_val: the value to compute the hessian against
    :returns: first gradient, eigen values and eigen vectors of the negative
    hessian inverse
    """
    first_gradient, hessian = tensorops.gradients(score, node_val)
    is_valid_first_grad_and_hessian = is_valid(first_gradient) or is_valid(hessian)
    if not is_valid_first_grad_and_hessian:
        return False, tensor(0.0), tensor(0.0), tensor(0.0)
    neg_hessian = -1 * hessian.detach()
    eig_vecs, eig_vals = soft_abs_inverse(neg_hessian)
    return True, first_gradient, eig_vecs, eig_vals


def hessian_of_log_prob(
    world: World,
    node: RVIdentifier,
    transformed_node_val: torch.Tensor,
    hessian_fn: Callable,
    transform: dist.Transform = dist.identity_transform,
) -> Tuple[torch.Tensor, torch.Tensor]:
    y = transformed_node_val.clone()
    y.requires_grad = True
    x = transform.inv(y)
    world_with_grad = world.replace({node: x})
    children = world_with_grad.get_variable(node).children
    score = (
        world_with_grad.log_prob(children | {node})
        - transform.log_abs_det_jacobian(x, y).sum()
    )
    return hessian_fn(score, y)
