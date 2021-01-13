import itertools
import logging
from typing import Callable, Optional

import flowtorch.bijectors
import flowtorch.params
import torch
import torch.distributions as dist
import torch.distributions.constraints as constraints
import torch.optim
from torch import Tensor
from torch.distributions.constraint_registry import biject_to


LOGGER = logging.getLogger("beanmachine.vi")


class VariationalApproximation(dist.distribution.Distribution):
    """
    Variational approximations on R^d.

    A variational approximation is a probability distribution with
    variational parameters which are optimized to maximize ELBO. This
    class encapsulates both the training (ELBO maximization against a
    target density's log_prob) as well as the resulting artifact (a
    probability distribution).
    """

    _transform: Optional[dist.transforms.Transform]

    def __init__(
        self,
        lr=1e-2,
        event_shape=torch.Size([]),  # noqa: B008
        base_dist=dist.Normal,
        base_args=None,
    ):
        """
        Construct a new VariationalApproximation.

        :params lr: learning rate
        :params base_dist: function (base_args) -> torch.distribution.Distribution
                           supporting `rsample`, used to initialize base distribution
        :param base_args: dict of kwargs for `base_dist`; any `nn.Parameter`s here will
                          be included in ELBO optimization
        """
        if not base_args:
            base_args = {}
        assert (
            len(base_dist(**base_args).event_shape) <= 1
        ), "VariationalApproximation only supports 0D/1D distributions"
        super(VariationalApproximation, self).__init__()

        # form independent product distribution of `base_dist` for `event_shape`
        _base_args = base_args
        _base_dist = base_dist
        if len(event_shape) == 1:
            d = event_shape[0]
            _base_args = {
                k: torch.nn.Parameter(torch.ones(d) * base_args[k]) for k in base_args
            }

            def _base_dist(**kwargs):
                return dist.Independent(
                    base_dist(**kwargs),
                    1,
                )

        self.flow = flowtorch.bijectors.AffineAutoregressive(
            flowtorch.params.DenseAutoregressive()
        )
        self.base_args = _base_args
        self.base_dist = _base_dist(**self.base_args)
        self.new_dist, self._flow_params = self.flow(self.base_dist)
        self.optim = torch.optim.Adam(
            self.parameters(),
            lr=lr,
        )
        self.has_rsample = True
        self._transform = None

    def elbo(
        self,
        target_log_prob: Callable[[Tensor], Tensor],
        target_support: constraints.Constraint = constraints.real,
        num_elbo_mc_samples: int = 100,
    ) -> Tensor:
        "Monte-Carlo approximate the ELBO against `self.target_log_prob`"
        z0 = self.base_dist.rsample(sample_shape=(num_elbo_mc_samples, 1))
        zk = self.flow._forward(z0, self._flow_params)
        ldj = self.flow._log_abs_det_jacobian(z0, zk, self._flow_params)

        if target_support != constraints.real:
            self._transform = biject_to(target_support)
            zk_constr = self._transform(zk)  # pyre-ignore[29]
            ldj += self._transform.log_abs_det_jacobian(  # pyre-ignore[16]
                zk, zk_constr
            ).squeeze()
            zk = zk_constr

        # cross-entropy H(Q,P)
        obj = target_log_prob(zk).sum()

        # negative entropy -H(Q)
        obj -= self.base_dist.log_prob(z0).sum() - ldj.sum()

        # normalize by batch size
        obj /= num_elbo_mc_samples

        return obj

    def rsample(self, sample_shape=None):
        if not sample_shape:
            sample_shape = torch.Size()
        xs = self.new_dist.sample(sample_shape=sample_shape)
        if self._transform:
            xs = self._transform(xs)
        return xs

    def parameters(self):
        return itertools.chain(
            self._flow_params.parameters(),
            filter(lambda x: x.requires_grad, self.base_args.values()),
        )

    def log_prob(self, value):
        if not self._transform:
            return self.new_dist.log_prob(value)
        else:
            value_inv = self._transform.inv(value)
            log_prob = self.new_dist.log_prob(value_inv)
            log_prob -= self._transform.log_abs_det_jacobian(value_inv, value).squeeze()
            return log_prob
