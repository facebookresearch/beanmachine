import logging
from typing import Callable

import torch
import torch.distributions as dist
import torch.optim
from torch import Tensor

from ...model.utils import LogLevel
from .iaf import FlowStack


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

    def __init__(
        self,
        num_flows=8,
        lr=1e-2,
        base_dist=dist.Normal,
        base_args=None,
    ):
        """
        Construct a new VariationalApproximation.

        :param num_flows: number of flow layers
        :params lr: learning rate
        :params base_dist: function (base_args) -> torch.distribution.Distribution
                           supporting `rsample`, used to initialize base distribution
        :param base_args: dict of kwargs for `base_dist`; any `nn.Parameter`s here will
                          be included in ELBO optimization
        """
        if not base_args:
            base_args = {}
        assert len(base_dist(**base_args).event_shape) <= 1
        super(VariationalApproximation, self).__init__()
        self.flow_stack = FlowStack(
            num_flows=num_flows, base_dist=base_dist, base_args=base_args
        )
        self.optim = torch.optim.Adam(
            self.parameters(),
            lr=lr,
        )
        self.has_rsample = True

    def arg_constraints(self):
        # TODO(fixme)
        return {}

    def elbo(
        self,
        target_log_prob: Callable[[Tensor], Tensor],
        num_elbo_mc_samples: int = 100,
    ) -> Tensor:
        "Monte-Carlo approximate the ELBO against `self.target_log_prob`"
        z0, zk, ldj = self.flow_stack(shape=(num_elbo_mc_samples,))
        n, d = z0.shape

        # negative entropy -H(Q)
        obj = -(
            self.flow_stack.base_dist(**self.flow_stack.base_args).log_prob(z0).sum()
        )
        obj += ldj.sum()  # change of variable zk -> z0

        # cross-entropy H(Q,P)
        obj += target_log_prob(zk).sum()

        # normalize by batch size
        obj /= num_elbo_mc_samples

        return obj

    def train(
        self,
        target_log_prob: Callable[[Tensor], Tensor],
        epochs: int = 100,
        num_elbo_mc_samples: int = 100,
    ) -> "VariationalApproximation":
        """
        Trains the VariationalApproximation.

        ELBO against `target_log_prob` is approximated
        using `num_elbo_mc_samples` from the base distribution and
        optimized with respect to `nn.Parameter`s in `flow_stack`
        and `base_args.values()`.

        :param epochs: num optimization steps
        :param num_elbo_mc_samples: num MC samples to use for estimating ELBO
        """
        optim = self.optim
        for _ in range(epochs):
            loss = -self.elbo(target_log_prob, num_elbo_mc_samples)
            if not torch.isnan(loss):
                loss.backward(retain_graph=True)
                optim.step()
                optim.zero_grad()
            else:
                # TODO: caused by e.g. negative scales in `dist.Normal`;
                # fix using pytorch's `constraint_registry` to account for
                # `Distribution.arg_constraints`
                LOGGER.log(LogLevel.INFO, "Encountered NaNs in loss, skipping epoch")
        return self

    def rsample(self, sample_shape=None):
        if not sample_shape:
            sample_shape = torch.Size()
        _, xs, _ = self.flow_stack(shape=sample_shape)
        return xs

    def parameters(self):
        return list(self.flow_stack.parameters()) + list(
            filter(lambda x: x.requires_grad, self.flow_stack.base_args.values())
        )

    def log_prob(self, value):
        z = value
        ldj = 0.0
        for flow in reversed(self.flow_stack.flow):
            h = torch.zeros(flow.context_size)
            s_t = flow.s_t(z, h) + 1.5
            sigma_t = torch.sigmoid(s_t)
            m_t = flow.m_t(z, h)
            ldj += flow.log_det_jac(sigma_t)
            z_prev = (z - (1 - sigma_t) * m_t) / sigma_t
            z = z_prev
        return (
            self.flow_stack.base_dist(**self.flow_stack.base_args).log_prob(z).squeeze()
            - ldj
        )
