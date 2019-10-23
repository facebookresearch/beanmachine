# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from torch import Tensor


"""
Common statistic functions, they all get a Tensor as input and return a Tensor
as output
"""


def mean(query_samples: Tensor) -> Tensor:
    return query_samples.mean(dim=0)


def std(query_samples: Tensor) -> Tensor:
    return torch.std(query_samples, dim=0)


def confidence_interval(query_samples: Tensor) -> Tensor:
    percentile_list = [2.5, 50, 97.5]
    return torch.tensor(
        np.percentile(query_samples.detach().numpy(), percentile_list, axis=0)
    )


def _compute_r_hat(query_samples: Tensor) -> Tensor:
    n_chains, n_samples = query_samples.shape[:2]
    per_chain_avg = query_samples.mean(1)
    b = n_samples * torch.var(per_chain_avg, dim=0)
    w = torch.mean(torch.var(query_samples, dim=1), dim=0)
    var_hat = (n_samples - 1) / n_samples * w + (1 / n_samples) * b
    return torch.sqrt(var_hat / w)


def r_hat(query_samples: Tensor) -> Tensor:
    n_chains = query_samples.shape[0]
    if n_chains < 2:
        raise ValueError("r_hat cannot be computed with fewer than two chains")
    return _compute_r_hat(query_samples)


def split_r_hat(query_samples: Tensor) -> Tensor:
    n_chains, n_samples = query_samples.shape[:2]
    if n_chains < 2:
        raise ValueError("split_r_hat cannot be computed with fewer than two chains")
    n_chains = n_chains * 2
    n_samples = n_samples // 2
    query_samples = torch.cat(torch.split(query_samples, n_samples, dim=1)[0:2])
    return _compute_r_hat(query_samples)
