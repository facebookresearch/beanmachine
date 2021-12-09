# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import beanmachine.ppl as bm
import gpytorch.likelihoods as likelihoods
import torch
from beanmachine.ppl.model.rv_identifier import RVIdentifier


class GpytorchMixin(torch.nn.Module):
    """
    Wrapper that registers the ``forward()`` call of GPyTorch likelihoods
    with Bean Machine. Priors for likelihood parameters can be registered
    similarly, as with kernels::

       from bm.experimental.likelihoods import GaussianLikelihood

       @bm.random_variable
       def noise_prior():
           return dist.Uniform(torch.tensor(0.01), torch.tensor(1.))

       likelihood = GaussianLikelihood(noise_prior=noise_prior)
       gp = SimpleGP(...)
       gp_prior = partial(gp, train_x)  # bind gp prior to train data
       obs = {likelihood(gp_prior): train_y}
       samples = nuts.infer([noise_prior()], obs, num_samples=100)
    """

    def _validate_args(self, prior):
        assert isinstance(prior(), RVIdentifier)
        "Prior should be None or a random variable but was: {}".format(type(prior))

    def __init__(self, *args, **kwargs):
        self.priors = {}
        for k, v in kwargs.copy().items():
            if "prior" not in k:
                continue

            self._validate_args(v)
            self.priors[k] = v

            # remove the prior for GPytorch
            kwargs.pop(k)
        super().__init__(*args, **kwargs)

    def __call__(self, prior_sample, *args, **kwargs):
        """
        In train mode, returns a sample from the likelihood given a `bm.random_variable`
        wrapped function. In eval mode, generates a marginal predictive sample.
        See `~gpytorch.likelihoods.Likelihood`.

        :param prior_sample: In train mode, a BM random variable. In eval mode, a
        `~gpytorch.distributions.MultivariateNormal` object.
        """
        if self.training:
            return self._bm_forward(prior_sample)
        return super().__call__(prior_sample, *args, **kwargs)

    @bm.random_variable
    def _bm_forward(self, prior_sample):
        return super().__call__(prior_sample())

    def train(self, mode=True):
        """
        In `train()` mode, parameters and the forward method are
        lifted to BM random variables.
        In `eval()` mode, this acts as a Gpytorch likelihood, ie
        all methods conform to the parent class's function signatures.
        """
        if mode:
            self._strict(True)
            if hasattr(self, "_priors"):
                self.priors = self._priors
            super().train()
        else:
            self._strict(False)
            self._priors = copy.deepcopy(self.priors)
            self.priors = {}
            super().train(False)

    @property
    def noise(self):
        if "noise_prior" in self.priors:
            return self.priors["noise_prior"]()
        return super().noise

    @noise.setter
    def noise(self, val):
        self.noise_covar.initialize(noise=val)

    @property
    def mixing_weights(self):
        if "mixing_weights_prior" in self.priors:
            return self.priors["mixing_weights_prior"]()
        return super().mixing_weights

    @property
    def scale(self):
        if "scale_prior" in self.priors:
            return self.priors["scale_prior"]()
        return super().scale

    @scale.setter
    def scale(self, value):
        self.initialize(raw_scale=self.raw_scale_constraint.inverse_transform(value))

    @property
    def task_noise_covar_factor(self):
        if "task_prior" in self.priors:
            return self.priors["task_prior"]()
        return super().task_noise_covar_factor

    @property
    def deg_free(self):
        if "deg_free_prior" in self.priors:
            return self.priors["deg_free_prior"]()
        return super().deg_free

    @deg_free.setter
    def deg_free(self, value):
        self._set_deg_free(value)


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
