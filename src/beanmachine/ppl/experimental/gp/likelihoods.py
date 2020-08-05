# Copyright (c) Facebook, Inc. and its affiliates.
from functools import partial

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
        A `random_variable` annotated callable. Returns a pointer to the callable
        that returns a torch distribution
        """
        fn = partial(super().forward, prior_sample())
        return fn(*args, **kwargs)

    @bm.random_variable
    def marginal(self, *args, **kwargs):
        # TODO this will work after predictive sampling is implemented
        return super().marginal(*args, **kwargs)


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
