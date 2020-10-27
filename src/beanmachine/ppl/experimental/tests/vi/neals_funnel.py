import torch
import torch.distributions as dist
from torch.distributions.utils import _standard_normal


class NealsFunnel(dist.distribution.Distribution):
    """
    Neal's funnel.

    p(x,y) = N(y|0,3) N(x|0,exp(y/2))
    """

    def __init__(self, validate_args=None):
        d = 2
        batch_shape, event_shape = (1,), (d,)
        super(NealsFunnel, self).__init__(
            batch_shape, event_shape, validate_args=validate_args
        )

    def rsample(self, sample_shape=torch.Size()):
        eps = _standard_normal(
            sample_shape, dtype=torch.float, device=torch.device("cpu")
        )
        z = torch.zeros(eps.shape)
        z[..., 1] = torch.tensor(3.0) * eps[..., 1]
        z[..., 0] = torch.exp(z[..., 1] / 2.0) * eps[..., 0]
        return z

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        x = value[..., 0]
        y = value[..., 1]

        log_prob = dist.Normal(0, 3).log_prob(y)
        log_prob += dist.Normal(0, torch.exp(y / 2)).log_prob(x)

        return log_prob
