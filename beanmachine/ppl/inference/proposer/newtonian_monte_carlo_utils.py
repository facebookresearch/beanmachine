from typing import Tuple

import torch
import torch.tensor as tensor
from torch import Tensor
from torch.autograd import grad


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
    if hasattr(node_val, "grad") and node_val.grad is not None:
        node_val.grad.zero_()


def compute_first_gradient(score: Tensor, node_val: Tensor) -> Tuple[bool, Tensor]:
    """
    Computes the first gradient.

    :param score: the score to compute the gradient of
    :param node_val: the value to compute the gradient against
    :returns: the first gradient
    """
    if not hasattr(node_val, "grad"):
        raise ValueError("requires_grad_ needs to be set for node values")

    # pyre expects attributes to be defined in constructors or at class
    # top levels and doesn't support attributes that get dynamically added.
    # pyre-fixme
    elif node_val.grad is not None:
        node_val.grad.zero_()

    first_gradient = grad(score, node_val, create_graph=True)[0]
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


def symmetric_inverse(neg_hessian: Tensor, max_zval: float = 1e5) -> Tensor:
    """
    Compute inverse of a symmetric matrix and returns inverse, eigen values
    and eigen vectors.

    :param neg_hessian: the value that we'd like to compute the inverse of
    :returns: neg_hessian inverse
    """
    eig_vals, eig_vecs = torch.eig(neg_hessian, eigenvectors=True)
    eig_vals = eig_vals[:, 0]
    zevals = eig_vals > max_zval
    neg_evals = eig_vals <= 0
    if torch.any(zevals) or torch.any(neg_evals):
        eig_vals[zevals] = max_zval
        eig_vals[neg_evals] = max_zval
    neg_hessian_inverse = eig_vecs * eig_vals.reciprocal().unsqueeze(0) @ eig_vecs.t()
    neg_hessian_inverse = (neg_hessian_inverse + neg_hessian_inverse.T) / 2
    # pyre-fixme[7]: Expected `Tensor` but got `float`.
    return neg_hessian_inverse


def compute_neg_hessian_invserse(
    first_gradient: Tensor,
    node_val: Tensor,
    min_diag_val: float = 1e-7,
    min_eig_val: float = 1e-5,
) -> Tuple[bool, Tensor]:
    """
    Compute negative hessian inverse.

    :param first_gradient: the first gradient of score with respect to
    node_val
    :param node_val: the value to compute the hessian against
    :returns: computes negative hessian inverse
    """
    is_valid, hessian = compute_hessian(first_gradient, node_val)
    if not is_valid:
        return False, tensor(0.0)
    # to avoid problems with inverse, here we add a small value - 1e-7 to
    # the diagonals
    diag = min_diag_val * torch.eye(hessian.shape[0])
    neg_hessian = -1 * (hessian + diag)
    # pyre-fixme
    neg_hessian_inverse = neg_hessian.inverse()
    eig_vals, eig_vec = torch.eig(neg_hessian_inverse, eigenvectors=True)
    eig_vals = eig_vals[:, 0]
    num_neg_eig_vals = (eig_vals < 0).sum()
    if num_neg_eig_vals.item() > 0:
        eig_vals[eig_vals < 1e-5] = min_eig_val
        eig_vals = torch.eye(len(eig_vals)) * eig_vals
        eig_vals_64 = eig_vals.to(dtype=torch.float64)
        eig_vec_64 = eig_vec.to(dtype=torch.float64)
        neg_hessian_inverse = eig_vec_64 @ eig_vals_64 @ eig_vec_64.T
        if eig_vals.dtype is torch.float32:
            neg_hessian_inverse = neg_hessian_inverse.to(dtype=torch.float32)
    return True, neg_hessian_inverse
