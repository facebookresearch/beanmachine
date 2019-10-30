# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

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


def _compute_var(query_samples: Tensor) -> Tuple[Tensor, Tensor]:
    n_chains, n_samples = query_samples.shape[:2]
    if n_chains > 1:
        per_chain_avg = query_samples.mean(1)
        b = n_samples * torch.var(per_chain_avg, dim=0)
    else:
        b = 0
    w = torch.mean(torch.var(query_samples, dim=1), dim=0)
    var_hat = (n_samples - 1) / n_samples * w + (1 / n_samples) * b
    return w, var_hat


def r_hat(query_samples: Tensor) -> Tensor:
    n_chains = query_samples.shape[0]
    if n_chains < 2:
        raise ValueError("r_hat cannot be computed with fewer than two chains")
    w, var_hat = _compute_var(query_samples)
    return torch.sqrt(var_hat / w)


def split_r_hat(query_samples: Tensor) -> Tensor:
    n_chains, n_samples = query_samples.shape[:2]
    if n_chains < 2:
        raise ValueError("split_r_hat cannot be computed with fewer than two chains")
    n_chains = n_chains * 2
    n_samples = n_samples // 2
    query_samples = torch.cat(torch.split(query_samples, n_samples, dim=1)[0:2])
    w, var_hat = _compute_var(query_samples)
    return torch.sqrt(var_hat / w)


def effective_sample_size(query_samples: Tensor) -> Tensor:
    n_chains, n_samples, *query_dim = query_samples.shape

    samples = query_samples - query_samples.mean(dim=1, keepdim=True)
    samples = samples.transpose(1, -1)
    # computes fourier transform (with padding)
    padded_samples = torch.cat((samples, torch.zeros(samples.shape)), dim=-1)
    fvi = torch.rfft(padded_samples, 1, onesided=False)
    # multiply by complex conjugate
    acf = fvi.pow(2).sum(-1, keepdim=True)
    # transform back to reals (with padding)
    padded_acf = torch.cat((acf, torch.zeros(acf.shape)), dim=-1)
    rho_per_chain = torch.irfft(padded_acf, 1, onesided=False)

    rho_per_chain = rho_per_chain.narrow(-1, 0, n_samples)
    rho_per_chain = rho_per_chain / (torch.tensor(range(n_samples, 0, -1)))
    rho_per_chain = rho_per_chain.transpose(1, -1)

    rho_avg = rho_per_chain.mean(dim=0)
    w, var_hat = _compute_var(query_samples)
    if n_chains > 1:
        rho = 1 - ((w - rho_avg) / var_hat)
    else:
        rho = rho_avg / var_hat
    rho[0] = 1

    # reshape to 2d matrix where each row contains all samples for specific dim
    rho_2d = torch.stack(torch.unbind(rho, dim=0), dim=-1).reshape(-1, n_samples)
    rho_sum = torch.zeros(rho_2d.shape[0])

    for i, chain in enumerate(torch.unbind(rho_2d, dim=0)):
        total_sum = torch.tensor(0.0)
        for t in range(n_samples // 2):
            rho_even = chain[2 * t]
            rho_odd = chain[2 * t + 1]
            if rho_even + rho_odd < 0:
                break
            else:
                total_sum += rho_even + rho_odd
        rho_sum[i] = total_sum

    rho_sum = torch.reshape(rho_sum, query_dim)
    return torch.div(n_chains * n_samples, -1 + 2 * rho_sum)
