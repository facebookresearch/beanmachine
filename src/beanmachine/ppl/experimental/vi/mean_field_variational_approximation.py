# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
from typing import Callable, Dict

import flowtorch.bijectors
import flowtorch.distributions
import flowtorch.parameters
import torch
import torch.distributions as dist
import torch.distributions.constraints as constraints
import torch.optim
from torch import Tensor
from torch.distributions.constraint_registry import biject_to, transform_to
from torch.distributions.utils import _sum_rightmost


LOGGER = logging.getLogger("beanmachine.vi")


class MeanFieldVariationalApproximation(dist.distribution.Distribution):
    """
    Mean field variational approximations on R^d.

    A variational approximation is a probability distribution with
    variational parameters which are optimized to maximize ELBO. Mean field
    approximations construct an independent non-interacting approximation for
    each `random_variable` site. This class encapsulates both the training
    (ELBO maximization against a target density's log_prob) as well as the
    resulting artifact (a probability distribution).
    """

    _transform: dist.transforms.Transform
    base_arg_constraints: Dict[str, dist.constraints.Constraint]

    def __init__(
        self,
        target_dist: dist.Distribution,
        lr=1e-2,
        base_dist=dist.Normal,
        base_args=None,
        validate_args=None,
    ):
        """
        Construct a new MeanFieldVariationalApproximation.

        :params lr: learning rate
        :params base_dist: function (base_args) -> torch.distribution.Distribution
                           supporting `rsample`, used to initialize base distribution
        :param base_args: dict of kwargs for `base_dist`; any `nn.Parameter`s here will
                          be included in ELBO optimization
        """
        self.base_arg_constraints = base_dist.arg_constraints
        assert (
            base_dist.has_rsample
        ), "The base distribution used for mean-field variational inference must support reparameterized sampling"
        self.has_rsample = True

        if not base_args:
            base_args = {}

        event_shape = target_dist.event_shape  # pyre-ignore[16]
        # form independent product distribution of `base_dist` for `event_shape`
        if len(event_shape) == 0:
            self.base_args = base_args
            self.base_dist = base_dist
        elif len(event_shape) <= 1:
            d = event_shape[0]
            self.base_args = {
                k: torch.nn.Parameter(torch.ones(d) * base_args[k]) for k in base_args
            }

            def _base_dist(**kwargs):
                return dist.Independent(
                    base_dist(**kwargs),
                    1,
                )

            self.base_dist = _base_dist

        else:
            raise NotImplementedError(
                "MeanFieldVariationalApproximation only supports 0D/1D distributions"
            )

        # TODO: remove once fixed upstream
        _tmp = flowtorch.parameters.DenseAutoregressive()

        # TODO: This needs to be now set after self.new_dist has been instantiated
        # _tmp._init_weights = lambda layers: None
        self.flow = flowtorch.bijectors.AffineAutoregressive(params=_tmp)

        assert len(base_dist(**base_args).event_shape) <= 1
        self.new_dist = flowtorch.distributions.Flow(
            self.recompute_transformed_distribution(), self.flow
        )
        self._flow_params = self.new_dist.bijector.params
        self.optim = torch.optim.Adam(
            self.parameters(),
            lr=lr,
        )

        # unwrap nested independents before setting transform
        support = target_dist.support  # pyre-ignore[16]
        while isinstance(support, constraints.independent):
            support = support.base_constraint
        self._transform = biject_to(support)

        super().__init__(
            self.new_dist.batch_shape,  # pyre-ignore
            self.new_dist.event_shape,  # pyre-ignore
            validate_args=validate_args,
        )

    def recompute_transformed_distribution(self):
        """
        Recomputes the flow's base distribution after `transform_to(arg_constraints)`
        is applied to the base distribution's arguments.
        """
        flow_base_dist = self.base_dist(
            **{
                k: transform_to(self.base_arg_constraints[k])(v)
                for k, v in self.base_args.items()
            }
        )
        if hasattr(self, "new_dist"):
            self.new_dist.base_dist = flow_base_dist

        return flow_base_dist

    def elbo(
        self,
        target_log_prob: Callable[[Tensor], Tensor],
        num_elbo_mc_samples: int = 100,
    ) -> Tensor:
        "Monte-Carlo approximate the ELBO against `self.target_log_prob`"
        zk = self.rsample(sample_shape=(num_elbo_mc_samples,))
        elbo = target_log_prob(zk).sum()
        elbo -= self.log_prob(zk).sum()

        # normalize by batch size
        elbo /= num_elbo_mc_samples
        return elbo

    def rsample(self, sample_shape=None):
        if not sample_shape:
            sample_shape = torch.Size()
        xs = self.new_dist.rsample(sample_shape=sample_shape)
        xs = self._transform(xs)
        return xs

    def parameters(self):
        return itertools.chain(
            self._flow_params.parameters(),
            filter(lambda x: x.requires_grad, self.base_args.values()),
        )

    def log_prob(self, value):
        value_inv = self._transform.inv(value)

        # TODO: do we need to a _sum_rightmost here?
        log_prob = self.new_dist.log_prob(value_inv)
        log_prob -= _sum_rightmost(
            self._transform.log_abs_det_jacobian(value_inv, value),
            len(self.new_dist.event_shape),
        )

        return log_prob
