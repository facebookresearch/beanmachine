# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
import torch.autograd


def gradients(
    outputs: torch.Tensor, inputs: torch.Tensor, allow_unused: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the first and the second gradient of the output Tensor
    w.r.t. the input Tensor.

    :param output: A Tensor variable with a single element.
    :param input: A 1-d tensor input variable that was used to compute the
                output. Note: the input must have requires_grad=True
    :returns: tuple of Tensor variables -- The first and the second gradient.
    """
    if outputs.numel() != 1:
        raise ValueError(
            f"output tensor must have exactly one element, got {outputs.numel()}"
        )

    grad1 = torch.autograd.grad(
        outputs, inputs, create_graph=True, retain_graph=True, allow_unused=allow_unused
    )[0].reshape(-1)

    # using identity matrix to reconstruct the full hessian from vector-Jacobian product
    hessians = torch.vmap(
        lambda vec: torch.autograd.grad(
            grad1,
            inputs,
            vec,
            create_graph=True,
            retain_graph=True,
            allow_unused=allow_unused,
        )[0].reshape(-1)
    )(torch.eye(grad1.size(0)))
    return grad1, hessians


def halfspace_gradients(
    outputs: torch.Tensor, inputs: torch.Tensor, allow_unused: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the first and the second gradient of the output Tensor w.r.t. the input
     Tensor for half space.

    :param output: A Tensor variable with a single element.
    :param input: A 1-d tensor input variable that was used to compute the
                output. Note: the input must have requires_grad=True
    :returns: tuple of Tensor variables -- The first and the second gradient.
    """
    grad1, hessians = gradients(outputs, inputs, allow_unused)
    return grad1, torch.diagonal(hessians)


def simplex_gradients(
    outputs: torch.Tensor, inputs: torch.Tensor, allow_unused: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the first and the second gradient of the output Tensor w.r.t. the input
     Tensor for simplex.

    :param output: A Tensor variable with a single element.
    :param input: A 1-d tensor input variable that was used to compute the
                output. Note: the input must have requires_grad=True
    :returns: tuple of Tensor variables -- The first and the second gradient.
    """
    grad1, hessians = gradients(outputs, inputs, allow_unused)
    hessian_diag = torch.diagonal(hessians).clone()
    # mask diagonal entries
    hessians[torch.eye(hessians.size(0)).bool()] = float("-inf")
    hessian_diag -= hessians.max(dim=0)[0]
    return grad1, hessian_diag
