# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
Reference paper:
    Lloyd, James, et al.
    "Automatic construction and natural-language description of nonparametric regression models."
    Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 28. No. 1. 2014.
    link: https://arxiv.org/pdf/1402.4304.pdf
"""

import torch
from gpytorch.constraints import Positive
from gpytorch.kernels import Kernel
from torch.nn import ModuleList  # @manual


class WhiteNoiseKernel(Kernel):
    """
    Computes a covariance matrix based on white noise.

    :param noise: the noise to add to diagnal.
    :param noise_prior: (Prior, optional)
        Set this if you want to apply a prior to the noise parameter.  Default: `None`.
    :param noise_constraint: (Constraint, optional)
        Set this if you want to apply a constraint to the noise parameter. Default: `Positive`.
    """

    def __init__(self, noise, noise_prior=None, noise_constraint=None, **kwargs):
        super(WhiteNoiseKernel, self).__init__(**kwargs)

        self.register_parameter(
            name="raw_noise",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)),
        )

        if noise_constraint is None:
            noise_constraint = Positive()

        if noise_prior is not None:
            self.register_prior(
                "noise_prior",
                noise_prior,
                lambda m: m.noise,
                lambda m, v: m._set_noise(v),
            )
        self.register_constraint("raw_noise", noise_constraint)
        self.noise = noise

    @property
    def noise(self):
        return self.raw_noise_constraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, value):
        self._set_noise(value)

    def _set_noise(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_noise)
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))

    def forward(self, x1, x2, **kwargs):
        res = torch.eye(x1.shape[0], x2.shape[0]) * self.noise
        return res


class ConstantKernel(Kernel):
    """
    Computes a covariance matrix which has constant values.

    :param constant: the constant to add to each element.
    :param constant_prior: (Prior, optional)
        Set this if you want to apply a prior to the constant parameter.  Default: `None`.
    :param constant_constraint: (Constraint, optional)
        Set this if you want to apply a constraint to the constant parameter. Default: `Positive`.
    """

    def __init__(
        self, constant, constant_prior=None, constant_constraint=None, **kwargs
    ):
        super(ConstantKernel, self).__init__(**kwargs)

        self.register_parameter(
            name="raw_constant",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)),
        )

        if constant_constraint is None:
            constant_constraint = Positive()

        if constant_prior is not None:
            self.register_prior(
                "constant_prior",
                constant_prior,
                lambda m: m.constant,
                lambda m, v: m._set_constant(v),
            )
        self.register_constraint("raw_constant", constant_constraint)
        self.constant = constant

    @property
    def constant(self):
        return self.raw_constant_constraint.transform(self.raw_constant)

    @constant.setter
    def constant(self, value):
        self._set_constant(value)

    def _set_constant(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_constant)
        self.initialize(
            raw_constant=self.raw_constant_constraint.inverse_transform(value)
        )

    def forward(self, x1, x2, **kwargs):
        res = torch.ones(*self.batch_shape, x1.shape[0], x2.shape[0]) * self.constant
        return res


class ChangePointKernel(Kernel):
    """
    Computes a covariance matrix that change from one kernel to another at various changepoint locations.

    :param kernels: a list of GPyTorch :class:`gpytorch.kernels.Kernel` instances. Must be one more
        than the number of change points.
    :param location: the location of the change points.
    :param steep: the steepness of sigmoid.
    :param location_prior: (Prior, optional)
        Set this if you want to apply a prior to the location parameter.  Default: `None`.
    :param location_constraint: (Constraint, optional)
        Set this if you want to apply a constraint to the location parameter.  Default: `GreaterThan(l1)`.
    :param steep_prior: (Prior, optional)
        Set this if you want to apply a prior to the steep parameter.  Default: `None`.
    :param steep_constraint: (Constraint, optional)
        Set this if you want to apply a constraint to the steep parameter. Default: `Positive`.
    """

    def __init__(
        self,
        kernels,
        location,
        steep=1.0,
        location_prior=None,
        location_constraint=None,
        steep_prior=None,
        steep_constraint=None,
        **kwargs,
    ):
        location = torch.as_tensor(location)
        steep = torch.as_tensor(steep)
        self.num_changepoints = 1 if location.numel() == 1 else location.shape[-1]
        if steep.shape != location.shape:
            steep = steep.expand(location.shape)
        assert len(kernels) == self.num_changepoints + 1
        super().__init__(**kwargs)
        self._kernels = ModuleList(kernels)

        self.register_parameter(
            name="raw_location",
            parameter=torch.nn.Parameter(
                torch.zeros(*self.batch_shape, 1, 1, self.num_changepoints)
            ),
        )

        if location_prior is not None:
            self.register_prior(
                "location_prior",
                location_prior,
                lambda m: m.location,
                lambda m, v: m._set_location(v),
            )
        if location_constraint is not None:
            self.register_constraint("raw_location", location_constraint)
        self.location = location

        self.register_parameter(
            name="raw_steep",
            parameter=torch.nn.Parameter(
                torch.ones(*self.batch_shape, 1, 1, self.num_changepoints)
            ),
        )

        if steep_prior is not None:
            self.register_prior(
                "steep_prior",
                steep_prior,
                lambda m: m.steep,
                lambda m, v: m._set_steep(v),
            )
        if steep_constraint is None:
            steep_constraint = Positive()
        self.register_constraint("raw_steep", steep_constraint)
        self.steep = steep

    @property
    def location(self):
        constraint = getattr(self, "raw_location_constraint", None)
        if constraint:
            return self.raw_location_constraint.transform(self.raw_location)
        return self.raw_location

    @location.setter
    def location(self, value):
        self._set_location(value)

    def _set_location(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_location)
        constraint = getattr(self, "raw_location_constraint", None)
        if constraint:
            self.initialize(
                raw_location=self.raw_location_constraint.inverse_transform(value)
            )
        else:
            self.initialize(raw_location=value)

    @property
    def steep(self):
        return self.raw_steep_constraint.transform(self.raw_steep)

    @steep.setter
    def steep(self, value):
        self._set_steep(value)

    def _set_steep(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_steep)
        self.initialize(raw_steep=self.raw_steep_constraint.inverse_transform(value))

    def _sigmoid(
        self, X: torch.Tensor, last_dim_is_batch: bool = False
    ) -> torch.Tensor:
        if last_dim_is_batch:
            location = self.location.unsqueeze(-2)
            steep = self.steep.unsqueeze(-2)
        else:
            location = self.location
            steep = self.steep
        return 0.5 + 0.5 * torch.tanh(steep * (X[..., None] - location))

    def num_outputs_per_input(self, x1, x2):
        return self._kernels[0].num_outputs_per_input(x1, x2)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)
        sig_x1 = self._sigmoid(x1, last_dim_is_batch)
        sig_x2 = self._sigmoid(x2, last_dim_is_batch)
        starters = sig_x1 * torch.transpose(sig_x2, -2, -3)
        stoppers = (1 - sig_x1) * torch.transpose((1 - sig_x2), -2, -3)
        ones = torch.ones(
            starters.shape[:-1] + (1,), dtype=starters.dtype, device=starters.device
        )
        starters = torch.cat([ones, starters], dim=-1)
        stoppers = torch.cat([stoppers, ones], dim=-1)

        kernel_stack = torch.stack(
            [k(x1, x2).evaluate() for k in self._kernels],
            dim=-1,
        )
        return torch.sum(kernel_stack * starters * stoppers, dim=-1)
