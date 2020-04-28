from typing import Optional, Tuple, Union

import torch
from beanmachine.ppl.utils import tensorops
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


def symmetric_inverse(
    neg_hessian: Tensor, max_zval: float = 1e8
) -> Tuple[Tensor, Tensor]:
    """
    Compute inverse of a symmetric matrix and returns inverse, eigen values
    and eigen vectors.

    :param neg_hessian: the value that we'd like to compute the inverse of
    :param max_zval: negative hessian eigen values will be set to max_zval and
    later inversed as we'd like to compute negative hessian inverse eigenvalues.
    :returns: eigen value and eigen vector of the negative hessian inverse
    """
    eig_vals, eig_vecs = torch.symeig(neg_hessian, eigenvectors=True)
    eig_vals[eig_vals <= 0] = max_zval
    inverse_eig_vals = eig_vals.reciprocal()
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
    # pyre-fixme
    first_gradient, hessian = tensorops.gradients(score, node_val)
    is_valid_first_grad_and_hessian = is_valid(first_gradient) or is_valid(hessian)
    if not is_valid_first_grad_and_hessian:
        return False, tensor(0.0), tensor(0.0), tensor(0.0)
    # to avoid problems with inverse, here we add a small value - 1e-7 to
    # the diagonals
    neg_hessian = -1 * hessian.detach()
    # pyre-fixme
    eig_vecs, eig_vals = symmetric_inverse(neg_hessian)
    return True, first_gradient, eig_vecs, eig_vals
