# Copyright (c) Facebook, Inc. and its affiliates.
import beanmachine.ppl as bm
import gpytorch.likelihoods as likelihoods


class GpytorchMixin(object):
    """
    Wrapper that registers the ``forward()`` call of GPyTorch likelihoods
    with Bean Machine
    """

    @bm.random_variable
    def forward(self, prior_sample, *args, **kwargs):
        """
        Returns a sample from the likelihood given a prior random variable.
        """
        return super().forward(prior_sample())


all_likelihoods = []
# Wrap all the likelihoods from GPytorch
for name, likelihood in likelihoods.__dict__.items():
    if not isinstance(likelihood, type):
        continue
    if not issubclass(likelihood, likelihoods.Likelihood):
        continue

    all_likelihoods.append(name)

    bm_likelihood = type(name, (GpytorchMixin, likelihood), {})
    bm_likelihood.__module__ = __name__
    locals()[name] = bm_likelihood
