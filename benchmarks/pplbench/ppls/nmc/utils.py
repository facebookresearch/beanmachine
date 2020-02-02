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
            raise AssertionError(
                "Hessian and gradient both positive for halfspace NMC proposer"
            )
    alpha = 1 - val ** 2 * hess
    beta = -val * hess - grad
    assert alpha > 0 and beta > 0
    return Gamma(alpha, beta)


def real_proposer(val, grad, hess):
    """
    propose for a multivariate real
    """
    # we will first pretend that everything is fine, but if we get an error
    # then we will try a more defensive approach to computing the proposal
    try:
        covar = (-hess).inverse()
        mu = val + (covar @ grad.unsqueeze(1)).squeeze(1)
        return MultivariateNormal(mu, covar)
    except RuntimeError:
        # fixup any negative eigenvalues
        evals, evecs = torch.eig(-hess, eigenvectors=True)
        evals = evals[:, 0]  # note: a symmetric matrix has only real eigen vals
        evals = evals ** -1  # we want eigen vals of the inverse
        evals[evals < 0] = 1e-8  # convert negative eigen vals to positive
        mu = val + (
            evecs @ (torch.eye(len(evals)) * evals) @ (evecs.T @ grad.unsqueeze(1))
        ).squeeze(1)
        return NormalEig(mu, evals, evecs)


class NormalEig(object):
    """
    A multivariate normal distribution where the covariance is specified
    through its eigen decomposition
    """

    def __init__(self, mean, eig_vals, eig_vecs):
        """
        mean - The mean of the multivariate normal.
        eig_vals - 1d vector of the eigen values (all positive) of the covar
        eig_vecs - 2d vector whose columns are the eigen vectors of the covar
        The covariance matrix of the multivariate normal is given by:
          eig_vecs @ (torch.eye(len(eig_vals)) * eig_vals) @ eig_vecs.T
        """
        assert mean.dim() == 1
        self.n = mean.shape[0]
        assert eig_vals.shape == (self.n,)
        assert eig_vecs.shape == (self.n, self.n)
        self.mean = mean
        self.eig_vecs = eig_vecs
        self.sqrt_eig_vals = eig_vals.sqrt().unsqueeze(0)
        # square root  of the covariance matrix
        self.sqrt_covar = self.sqrt_eig_vals * eig_vecs
        # log of sqrt of determinant
        self.log_sqrt_det = eig_vals.log().sum() / 2.0
        # a base distribution of independent normals is used to draw
        # samples that will be stretched along the eigen directions
        self.base_dist = torch.distributions.normal.Normal(
            torch.zeros(1, self.n), torch.ones(1, self.n)
        )

    def sample(self):
        with torch.no_grad():
            z = torch.normal(mean=0.0, std=1.0, size=(self.n, 1))
            return self.mean + (self.sqrt_covar @ z).squeeze(1)

    def log_prob(self, value):
        assert value.shape == (self.n,)
        z = ((value - self.mean).unsqueeze(0) @ self.eig_vecs) / self.sqrt_eig_vals
        return self.base_dist.log_prob(z).sum() - self.log_sqrt_det


def simplex_proposer(val, grad):
    """
    propose for a simplex-constrained random variable
    """
    conc = grad * val + 1
    assert (conc > 0).all()
    return Dirichlet(conc)
