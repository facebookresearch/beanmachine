# Copyright (c) Facebook, Inc. and its affiliates.
import torch


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
