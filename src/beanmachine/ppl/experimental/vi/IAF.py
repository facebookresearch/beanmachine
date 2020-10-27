import torch
import torch.nn as nn

from .MADE import AutoRegressiveNN


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
    def __init__(self, dim, n_flows):
        super().__init__()
        self.flow = nn.Sequential(*[IAF(dim) for _ in range(n_flows)])
        self.mu = nn.Parameter(torch.randn(dim,).normal_(0, 0.01))
        self.log_var = nn.Parameter(torch.randn(dim,).normal_(1, 0.01))

    def forward(self, shape):
        std = torch.exp(0.5 * self.log_var)
        eps = torch.randn(shape)  # unit gaussian
        z0 = self.mu + eps * std

        zk, ldj = self.flow({0: z0, 1: 0.0})
        return z0, zk, self.mu, self.log_var, ldj
