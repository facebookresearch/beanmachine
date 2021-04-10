import numbers
from collections import OrderedDict
from typing import Iterable, Optional, Set, Type

import torch
from gpytorch.kernels import (
    AdditiveKernel,
    Kernel,
    PeriodicKernel,
    ProductKernel,
    RBFKernel,
    ScaleKernel,
)
from sts.data import DataTensor
from torch import nn


class CovComponent(nn.Module):
    def __init__(self, kernel: Kernel, name=None):
        super().__init__()
        self.kernel = kernel
        self._name = name if name is not None else self._get_name()

    def __add__(self, other):
        components = []
        components += (
            list(self.parts.values())
            if isinstance(self, AdditiveComponents)
            else [self]
        )
        components += (
            list(other.parts.values())
            if isinstance(other, AdditiveComponents)
            else [other]
        )
        return AdditiveComponents(*components)

    def __mul__(self, other):
        components = []
        components += (
            list(self.parts.values()) if isinstance(self, ProductComponents) else [self]
        )
        components += (
            list(other.parts.values())
            if isinstance(other, ProductComponents)
            else [other]
        )
        return ProductComponents(*components)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def lengthscale(self):
        if isinstance(self.kernel, ScaleKernel):
            return self.kernel.base_kernel.lengthscale * self.scale
        return self.kernel.lengthscale * self.scale

    @property
    def scale(self):
        return 1.0

    @lengthscale.setter
    def lengthscale(self, value):
        if isinstance(self.kernel, ScaleKernel):
            self.kernel.base_kernel.lengthscale = value / self.scale
        else:
            self.kernel.lengthscale = value / self.scale

    @property
    def fixed_params(self) -> Set[nn.Parameter]:
        return set()

    def forward(self, x1, x2, diag=False, **params):
        return self.kernel.forward(x1, x2, diag=diag, **params)

    def __call__(self, x1, x2=None, diag=False, last_dim_is_batch=False, **params):
        if isinstance(x1, DataTensor):
            x1 = x1.tensor
        if isinstance(x2, DataTensor):
            x2 = x2.tensor
        return self.kernel(x1, x2, diag, last_dim_is_batch, **params)


class Regression(CovComponent):
    def __init__(
        self,
        sample_input: DataTensor,
        features: Iterable[str],
        *,
        ard: bool = True,
        kernel_cls: Type[Kernel] = RBFKernel,
        name: Optional[str] = None,
        **kernel_kwargs,
    ):
        if kernel_cls is ScaleKernel:
            raise ValueError(
                "Wrapping kernel in ScaleKernel is not required. Pass the base kernel instead."
            )
        self.features = features
        regression_kernel = ScaleKernel(
            kernel_cls(
                active_dims=sample_input.get_index(features),
                ard_num_dims=len(features) if ard else None,
                **kernel_kwargs,
            )
        )
        super().__init__(regression_kernel, name=name)
        scale = []
        for f in features:
            if f in sample_input.transforms:
                scale.append(sample_input.transforms.get(f, 1.0).inv.scale)
            else:
                scale.append(1.0)
        self._scale = torch.tensor(scale)

    @property
    def scale(self):
        return self._scale


class Trend(CovComponent):
    """
    Covariance component for modeling the trend.

    :param sample_input: a sample :class:`DataTensor` object that represents
        the inputs of interest.
    :param time_axis: the name for the time axis over which to model the
        trend.
    :param lengthscale: optional length scale to initialize to. Note that
        this is in the original (unnormalized) scale of the time unit.
    :param fix_lengthscale: whether to fix the `lengthscale`, i.e. remove
        this as a hyperparameter to be optimized.
    :param kernel_cls: a :class:`gpytorch.kernels.Kernel` class to be used
        for modeling trend; defaults to RBFKernel.
    :param name: optional name for the component.
    """

    def __init__(
        self,
        sample_input: DataTensor,
        time_axis: str,
        *,
        lengthscale: Optional[numbers.Real] = None,
        fix_lengthscale: bool = False,
        kernel_cls: Type[Kernel] = RBFKernel,
        name: Optional[str] = None,
        **kernel_kwargs,
    ):
        if kernel_cls is ScaleKernel:
            raise ValueError(
                "Wrapping kernel in ScaleKernel is not required. Pass the base kernel instead."
            )
        self.fix_lengthscale = fix_lengthscale and (lengthscale is not None)
        super().__init__(
            ScaleKernel(
                kernel_cls(
                    active_dims=(sample_input.get_index(time_axis),), **kernel_kwargs
                )
            ),
            name=name,
        )
        # normalize lengthscale if input data is normalized
        self._scale = 1.0
        if lengthscale is not None:
            if time_axis in sample_input.transforms:
                self._scale = sample_input.transforms[time_axis].inv.scale
            self.lengthscale = lengthscale

    @property
    def scale(self):
        return self._scale

    @property
    def fixed_params(self) -> Set[nn.Parameter]:
        return (
            {self.kernel.base_kernel.raw_lengthscale} if self.fix_lengthscale else set()
        )


class Seasonality(CovComponent):
    """
    Covariance component for modeling seasonality.

    :param sample_input: a sample :class:`DataTensor` object that represents
        the inputs of interest.
    :param time_axis: the name for the time axis over which to place the
        seasonality kernel.
    :param period_length: optional period length to initialize to. Note that
        this is in the original (unnormalized) scale of the time unit.
    :param fix_period: whether to fix the `period_length`, i.e. remove this as a
        hyperparameter to be optimized.
    :param name: optional name for the component.
    """

    def __init__(
        self,
        sample_input: DataTensor,
        time_axis: str,
        period_length: numbers.Real,
        *,
        lengthscale: Optional[int] = None,
        fix_period: bool = False,
        fix_lengthscale: bool = False,
        name: Optional[str] = None,
        **kernel_kwargs,
    ):
        self.fix_period = fix_period
        self.fix_lengthscale = fix_lengthscale
        super().__init__(
            ScaleKernel(
                PeriodicKernel(
                    active_dims=(sample_input.get_index(time_axis),), **kernel_kwargs
                )
            ),
            name=name,
        )
        # normalize lengthscale if input data is normalized
        self._scale = 1.0
        if time_axis in sample_input.transforms:
            self._scale = sample_input.transforms[time_axis].inv.scale
            period_length = period_length / self._scale
        self.kernel.base_kernel.period_length = period_length
        if lengthscale is not None:
            self.lengthscale = lengthscale

    @property
    def period_length(self):
        return self.kernel.base_kernel.period_length * self.scale

    @property
    def scale(self):
        return self._scale

    @property
    def fixed_params(self) -> Set[nn.Parameter]:
        fixed_params = set()
        if self.fix_period:
            fixed_params.add(self.kernel.base_kernel.raw_period_length)
        if self.fix_lengthscale:
            fixed_params.add(self.kernel.base_kernel.raw_lengthscale)
        return fixed_params


class Aggregated(CovComponent):
    def __init__(self, kernel_cls, *components, name=None):
        if name is None and components:
            # construct default name if components is non-empty
            name = f"{kernel_cls.__name__}({','.join(c.name for c in components)})"
        # Add components later so that their names are correctly populated
        super().__init__(kernel_cls(), name)
        self.parts = OrderedDict()
        for c in components:
            self.append(c)

    def append(self, component: CovComponent):
        name = component.name
        i = 1
        while name in self.parts:
            name = name + f"_{i}"
            i += 1
        component.name = name
        self.parts[name] = component
        self.kernel.kernels.append(component.kernel)

    @property
    def fixed_params(self) -> Set[nn.Parameter]:
        return set().union(*[k.fixed_params for k in self.parts.values()])


class ProductComponents(Aggregated):
    def __init__(self, *components, name=None):
        super().__init__(ProductKernel, *components, name=name)


class AdditiveComponents(Aggregated):
    def __init__(self, *components, name=None):
        super().__init__(AdditiveKernel, *components, name=name)


class Covariance(AdditiveComponents):
    """
    Represents the covariance function of the GP.

    :param sample_input: a sample :class:`DataTensor` object that represents
        the inputs of interest.
    """

    def __init__(self, sample_input: DataTensor):
        self.sample_input = sample_input
        super().__init__()

    # Some helper methods (can also simply self.append)
    def add_regression(
        self,
        features: Iterable[str],
        *,
        ard: bool = True,
        kernel_cls: Type[Kernel] = RBFKernel,
        name: Optional[str] = None,
        **kernel_kwargs,
    ):
        """
        Add a regression component to the model using an ARD kernel specified by
        the :class:`Regression` class.

        :param features: list of header names to be included as regressors.
        :param ard: Whether to enable Automatic Relevance Determination, i.e.
            have separate lengthscale for each of the features.
        :param kernel_cls: a :class:`gpytorch.kernels.Kernel` class to be used
            for regression.
        :param name: optional name for the component.
        """
        regression_kernel = Regression(
            self.sample_input,
            features,
            ard=ard,
            kernel_cls=kernel_cls,
            name=name,
            **kernel_kwargs,
        )
        self.append(regression_kernel)

    def add_seasonality(
        self,
        time_axis: str,
        period_length: numbers.Real,
        *,
        fix_period: bool = False,
        name: Optional[str] = None,
        **kernel_kwargs,
    ):
        """
        Add a seasonality component to the model using the :class:`Seasonality`
        class.

        :param time_axis: the name for the time axis over which to place the
            seasonality kernel.
        :param period_length: optional period length to initialize to. Note that
            this is in the original (unnormalized) scale of the time unit.
        :param fix_period: whether to fix the `period_length`, i.e. remove this
            as a hyperparameter to be optimized.
        :param name: optional name for the component.
        """
        seasonality_kernel = Seasonality(
            self.sample_input,
            time_axis,
            period_length,
            fix_period=fix_period,
            name=name,
            **kernel_kwargs,
        )
        self.append(seasonality_kernel)

    def add_trend(
        self,
        time_axis: str,
        *,
        lengthscale: Optional[numbers.Real] = None,
        fix_lengthscale: bool = False,
        kernel_cls: Type[Kernel] = RBFKernel,
        name: Optional[str] = None,
        **kernel_kwargs,
    ):
        """
        Add a trend component to the model using the :class:`Trend` class.

        :param time_axis: the name for the time axis over which to model the
            trend.
        :param lengthscale: optional length scale to initialize to. Note that
            this is in the original (unnormalized) scale of the time unit.
        :param fix_lengthscale: whether to fix the `lengthscale`, i.e. remove
            this as a hyperparameter to be optimized.
        :param kernel_cls: a :class:`gpytorch.kernels.Kernel` class to be used
            for modeling trend; defaults to RBFKernel.
        :param name: optional name for the component.
        """
        trend_kernel = Trend(
            self.sample_input,
            time_axis,
            lengthscale=lengthscale,
            fix_lengthscale=fix_lengthscale,
            kernel_cls=kernel_cls,
            name=name,
            **kernel_kwargs,
        )
        self.append(trend_kernel)
