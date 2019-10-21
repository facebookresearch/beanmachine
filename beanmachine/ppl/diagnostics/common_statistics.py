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
