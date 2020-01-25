# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.distributions import Dirichlet, Exponential, Gamma, MultivariateNormal


# python implementation of gradient computation
def py_gradients(output, inp):
    grad = torch.autograd.grad(output, inp, create_graph=True)[0]
    n = inp.numel()
    hess = torch.zeros(n, n)
    for j in range(n):
        hess[j] = torch.autograd.grad(grad[j], inp, retain_graph=True)[0]
    return grad, hess


# we will try to use a C++ implementation of gradients if possible
try:
    from beanmachine.ppl.utils import tensorops

    gradients = tensorops.gradients
except ImportError:
    print("Falling back to python implementation of gradients")
    gradients = py_gradients


def halfspace_proposer(val, grad, hess):
    """
    proposer for 1-d R+ variable
    """
    if hess > 0:
        if grad < 0:
            return Exponential(-grad)
        else:
            assert False
    alpha = 1 - val ** 2 * hess
    beta = -val * hess - grad
    assert alpha > 0 and beta > 0
    return Gamma(alpha, beta)


def real_proposer(val, grad, hess):
    """
    propose for a multivariate real
    """
    neg_hess_inv = torch.inverse(-hess)
    # fixup any negative eigenvalues
    eval, evec = torch.eig(neg_hess_inv, eigenvectors=True)
    eval = eval[:, 0]  # note: a symmetric matrix has only real eigen vals
    eval[eval < 0] = 1e-7  # convert negative eigen vals to positive
    covar = evec @ (torch.eye(len(eval)) * eval) @ evec.T
    mu = val + (neg_hess_inv @ grad.unsqueeze(1)).squeeze(1)
    return MultivariateNormal(mu, covar)


def simplex_proposer(val, grad):
    """
    propose for a simplex-constrained random variable
    """
    conc = grad * val + 1
    assert (conc > 0).all()
    return Dirichlet(conc)
