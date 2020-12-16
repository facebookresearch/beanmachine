import torch
import torch.distributions as dist
import torch.nn as nn

from .made import AutoRegressiveNN


class IAF(nn.Module):
    """
    Inverse Autoregressive Flow
    https://arxiv.org/pdf/1606.04934.pdf
    """

    def __init__(self, size=1, context_size=1, auto_regressive_hidden=1):
        super().__init__()
        self.context_size = context_size
        self.s_t = AutoRegressiveNN(
            in_features=size,
            hidden_features=auto_regressive_hidden,
            context_features=context_size,
        )
        self.m_t = AutoRegressiveNN(
            in_features=size,
            hidden_features=auto_regressive_hidden,
            context_features=context_size,
        )

    def log_det_jac(self, sigma_t):
        return torch.log(sigma_t + 1e-6).sum(1)

    def forward(self, args):
        z, ldj = args[0], args[1]
        h = torch.zeros(self.context_size)

        # Initially s_t should be large, i.e. 1 or 2.
        s_t = self.s_t(z, h) + 1.5
        sigma_t = torch.sigmoid(s_t)
        m_t = self.m_t(z, h)

        # transformation
        return sigma_t * z + (1 - sigma_t) * m_t, ldj + self.log_det_jac(sigma_t)


class FlowStack(nn.Module):
    "A (multi-layer) parametric flow transform of a base distribution."

    def __init__(self, num_flows, base_dist=dist.Normal, base_args=None):
        """
        Constructs a new `FlowStack`

        :param num_flows: number of flow layers
        :param base_dist: constructor fn for base distribution
        :param base_args: kwargs dict for base_dist
        """
        super().__init__()
        self.base_dist = base_dist
        if not base_args:
            base_args = {}
        self.base_args = base_args
        assert self.base_dist(**self.base_args).has_rsample
        dim = (
            base_dist(**base_args).event_shape[0]
            if len(base_dist(**base_args).event_shape) == 1
            else 1
        )
        self.flow = nn.Sequential(*[IAF(dim) for _ in range(num_flows)])

    def forward(self, shape):
        """
        Samples the base distribution and transforms samples using flow.

        :param shape: shape of the base distribution sample
        :return: the base distribution sample z0, the transformed sample zk,
                 and the log det jacobian
        """
        z0 = self.base_dist(**self.base_args).rsample(shape)
        if z0.ndim == 1:
            # ensure tensor is 2D
            z0 = z0.unsqueeze(1)

        zk, ldj = self.flow({0: z0, 1: 0.0})
        return z0, zk, ldj
