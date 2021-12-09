# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import gpytorch.kernels as kernels
import torch
from beanmachine.ppl.model.rv_identifier import RVIdentifier


class KernelMixin(torch.nn.Module):
    """
    Wrapper for Gpytorch kernels that lifts parameter attributes
    to Bean Machine `random_variable`s. This class can be freely composed
    with any GPytorch kernel. Any kernel with custom
    parameters (not defined in GPytorch) that will be inferred
    should extend this class and
    `~gpytorch.kernels.Kernel` and define parameters as properties
    that sample from the prior and a `forward()` method for
    computing the covariance matrix::

       @random_variable
       def prior():
           return dist.Normal(...)

       x = ScaleKernel(outputscale_prior=prior)
       x.lengthscale  # returns an invocation of the prior function
                      # (used during inference)

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

    def __call__(self, *args, **kwargs):
        out = super().__call__(*args, **kwargs)
        if self.training:
            return out.evaluate()
        return out

    def train(self, mode=True):
        if mode:
            # loads bm priors
            self._strict(True)
            if hasattr(self, "_priors"):
                self.priors = self._priors
            if hasattr(self, "base_kernel"):
                self.base_kernel.train()
            return super().train()
        else:
            # eval mode is a gpytorch kernel
            self._strict(False)
            self._priors = copy.deepcopy(self.priors)
            self.priors = {}
            if hasattr(self, "base_kernel"):
                self.base_kernel.eval()
            return super().train(False)

    @property
    def lengthscale(self):
        if "lengthscale_prior" in self.priors:
            return self.priors["lengthscale_prior"]()
        return super().lengthscale

    @lengthscale.setter
    def lengthscale(self, val):
        super()._set_lengthscale(val)

    @property
    def outputscale(self):
        if "outputscale_prior" in self.priors:
            return self.priors["outputscale_prior"]()
        return super().outputscale

    @outputscale.setter
    def outputscale(self, val):
        super()._set_outputscale(val)

    @property
    def variance(self):
        if "variance_prior" in self.priors:
            return self.priors["variance_prior"]()
        return super().variance

    @variance.setter
    def variance(self, val):
        super()._set_variance(val)

    @property
    def offset(self):
        if "offset_prior" in self.priors:
            return self.priors["offset_prior"]()
        return super().offset

    @offset.setter
    def offset(self, val):
        self._set_offset(val)

    @property
    def angular_weights(self):
        if "angular_weights_prior" in self.priors:
            return self.priors["angular_weights_prior"]()
        return super().angular_weights

    @angular_weights.setter
    def angular_weights(self, value):
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        super().initialize(
            raw_angular_weights=self.raw_angular_weights_constraint.inverse_transform(
                value
            )
        )

    @property
    def alpha(self):
        if "alpha_prior" in self.priors:
            return self.priors["alpha_prior"]()
        return super().alpha

    @alpha.setter
    def alpha(self, value):
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        super().initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))

    @property
    def beta(self):
        if "beta_prior" in self.priors:
            return self.priors["beta_prior"]()
        return super().beta

    @beta.setter
    def beta(self, value):
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        super().initialize(raw_beta=self.raw_beta_constraint.inverse_transform(value))

    @property
    def mixture_means(self):
        if "mixture_means_prior" in self.priors:
            return self.priors["mixture_means_prior"]()
        return super().mixture_means

    @mixture_means.setter
    def mixture_means(self, value):
        super()._set_mixture_means(value)

    @property
    def mixture_scales(self):
        if "mixture_scales_prior" in self.priors:
            return self.priors["mixture_scales_prior"]()
        return super().mixture_scales

    @mixture_scales.setter
    def mixture_scales(self, value):
        super()._set_mixture_scales(value)

    @property
    def mixture_weights(self):
        if "mixture_weights_prior" in self.priors:
            return self.priors["mixture_weights_prior"]()
        return super().mixture_weights

    @mixture_weights.setter
    def mixture_weights(self, value):
        super()._set_mixture_weights(value)

    @property
    def period_length(self):
        if "period_length_prior" in self.priors:
            return self.priors["period_length_prior"]()
        return super().period_length

    @period_length.setter
    def period_length(self, value):
        super()._set_period_length(value)

    @property
    def var(self):
        if "var" in self.priors:
            return self.priors["var"]()
        return super().var

    @var.setter
    def var(self, val):
        super()._set_var(val)


all_kernels = []
# Wrap all the kernels from GPytorch
for name, kernel in kernels.__dict__.items():
    if not isinstance(kernel, type):
        continue
    if not issubclass(kernel, kernels.Kernel):
        continue

    all_kernels.append(name)

    bm_kernel = type(name, (KernelMixin, kernel), {})
    bm_kernel.__module__ = __name__
    bm_kernel.__doc__ = """
    Wraps `{}`.{}` with `beanmachine.ppl.experimental.kernels.KernelMixin.
    """.format(
        bm_kernel.__module__, bm_kernel.__name__
    )
    locals()[name] = bm_kernel
