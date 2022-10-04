# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gpytorch as gpt
import torch
from botorch.models.gpytorch import GPyTorchModel
from botorch.posteriors.gpytorch import GPyTorchPosterior


class SimpleGP(gpt.models.ExactGP, GPyTorchModel):
    """
    GPytorch model that supports Bean Machine sampling and broadcasting semantics.
    In train mode, BM priors may be specified over GP parameters. In eval mode,
    this objects acts as a Gpytorch model and generates predictions using Gpytorch's
    prediction strategies.  For an example, see the [tutorial](link:TODO)
    """

    def __init__(self, x_train, y_train, mean, kernel, likelihood, *args, **kwargs):
        super().__init__(x_train, y_train, likelihood)
        self.mean = mean
        self.kernel = kernel

    def forward(self, data, *args, **kwargs):
        """
        Default forward definining a GP prior. Should be overridden by child class.
        """
        mean = self.mean(data)
        cov = self.kernel(data)
        return gpt.distributions.MultivariateNormal(mean, cov)

    def bm_load_samples(self, rv_dict):
        """
        Loads tensors from a dict keyed on module name and valued by tensor
        whose shape is (num_samples, sample_shape). See `~gpytorch.Module.initialize`.

        :param rv_dict: Python dict keyed on module name and valued by tensor
        whose shape is (num_samples, sample_shape)
        """
        self.pyro_load_from_samples(rv_dict)


class BoTorchGP(SimpleGP, GPyTorchModel):
    """
    Experimental module that is compatible with BoTorch.

    samples = nuts.infer(queries, obs, num_samples).get_chain(0)
    gp.eval()
    gp.bm_load_samples({kernel.lengthscale=samples[lengthscale_prior()]})

    from botorch.acquisition.objective import IdentityMCObjective
    acqf = get_acquisition_function("qEI", gp, IdentityMCObjective(), x_train)
    new_point = acqf(new_input).mean()
    """

    def __init__(self, x_train, y_train, *args, **kwargs):
        super().__init__(x_train, y_train, *args, **kwargs)
        if y_train.dim() > 1:
            self._num_outputs = y_train.shape[-1]
        else:
            self._num_outputs = 1

    def posterior(self, data, observation_noise=False, **kwargs):
        """
        Returns the posterior conditioned on new data. Used in BoTorch.
        See `~botorch.models.model.Model.posterior`.

        :param data: a `torch.Tensor` containing test data of shape `(batch, data_dim)`.
        :returns: `~botorch.posteriors.gpytorch.GPytorchPosterior` MultivariateNormal
        distribution.
        """
        self.eval()
        try:
            mvn = self(data, batch_shape=(data.shape[0],))
        except AttributeError as e:
            raise AttributeError(
                "Running in eval mode but one of the parameters is still"
                "a BM random variable. Did you `bm_load_samples`? \n" + str(e)
            )

        if observation_noise is not False:
            if torch.is_tensor(observation_noise):
                # TODO: Make sure observation noise is transformed correctly
                self._validate_tensor_args(X=data, Y=observation_noise)
                if observation_noise.shape[-1] == 1:
                    observation_noise = observation_noise.squeeze(-1)
                mvn = self.likelihood(mvn, data, noise=observation_noise)
            else:
                mvn = self.likelihood(mvn, data)
        return GPyTorchPosterior(mvn=mvn)
