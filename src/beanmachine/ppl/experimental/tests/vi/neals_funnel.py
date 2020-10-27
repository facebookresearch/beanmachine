import torch
import torch.distributions as dist
from torch.distributions.utils import _standard_normal


class NealsFunnel(dist.distribution.Distribution):
    def __init__(self, d=2, validate_args=None):
        batch_shape, event_shape = (1,), (d,)
        super(NealsFunnel, self).__init__(
            batch_shape, event_shape, validate_args=validate_args
        )

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=torch.float, device=torch.device("cpu"))
        z = torch.zeros(eps.shape)
        z[..., 1] = torch.sqrt(torch.tensor(3.0)) * eps[..., 1]
        z[..., 0] = torch.exp(z[..., 1] / 4.0) * eps[..., 0]
        return z

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        x = value[..., 0]
        y = value[..., 1]

        log_prob = dist.Normal(0, 3).log_prob(y)
        log_prob += dist.Normal(0, torch.exp(y / 2)).log_prob(x)

        return log_prob
