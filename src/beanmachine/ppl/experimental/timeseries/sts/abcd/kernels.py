# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from sts.gp.kernels import ChangePointKernel
from torch.nn import ModuleList  # @manual


class ChangeKernel(ChangePointKernel):
    @property
    def kernels(self):
        return list(dict.fromkeys(self._kernels))

    @kernels.setter
    def kernels(self, kernels):
        self._kernels = ModuleList(kernels)


class ChangePointABCDKernel(ChangeKernel):
    """
    Wrapper over :class:`sts.gp.kernels.ChangePointKernel` for use with the :mod:`sts.abcd` module.

    :param k1: a kernel.
    :param k2: a kernel.
    :param location: the location of change point.
    :param steep: the steepness of sigmoid.
    :param location_prior: (Prior, optional)
        Set this if you want to apply a prior to the location parameter.  Default: `None`.
    :param location_constraint: (Constraint, optional)
        Set this if you want to apply a constraint to the location parameter.  Default: `GreaterThan(location)`.
    :param steep_prior: (Prior, optional)
        Set this if you want to apply a prior to the steep parameter.  Default: `None`.
    :param steep_constraint: (Constraint, optional)
        Set this if you want to apply a constraint to the steep parameter. Default: `Positive`.
    """

    def __init__(
        self,
        k1,
        k2,
        location=0.0,
        steep=1.0,
        location_prior=None,
        location_constraint=None,
        steep_prior=None,
        steep_constraint=None,
        **kwargs,
    ):
        super().__init__(
            [k1, k2],
            location,
            steep,
            location_prior=location_prior,
            location_constraint=location_constraint,
            steep_prior=steep_prior,
            steep_constraint=steep_constraint,
            **kwargs,
        )


class ChangeWindowABCDKernel(ChangeKernel):
    """
    Wrapper over :class:`sts.gp.kernels.ChangePointKernel` for use with the :mod:`sts.abcd` module to simulate
    a changepoint window.

    :param k1: a kernel.
    :param k2: a kernel.
    :param location: the location of two change points location=(l1, l2). The second kernel functions in the window (l1, l2).
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
        k1,
        k2,
        location=(0.0, 3.0),
        steep=(1.0, 2.0),
        location_prior=None,
        location_constraint=None,
        steep_prior=None,
        steep_constraint=None,
        **kwargs,
    ):
        super().__init__(
            [k1, k2, k1],
            location,
            steep,
            location_prior=location_prior,
            location_constraint=location_constraint,
            steep_prior=steep_prior,
            steep_constraint=steep_constraint,
            **kwargs,
        )

    @property
    def kernels(self):
        return super().kernels

    @kernels.setter
    def kernels(self, kernels):
        self._kernels = ModuleList(kernels + [kernels[0]])
