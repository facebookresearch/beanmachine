# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, NamedTuple, Tuple

import numpy as np
import plotly.graph_objs as go
import torch
from torch import Tensor


class SamplesSummary(NamedTuple):
    num_chain: int
    num_samples: int
    single_sample_sz: Tensor


def _samples_info(query_samples: Tensor):
    return SamplesSummary(
        num_chain=query_samples.size(0),
        num_samples=query_samples.size(1),
        single_sample_sz=query_samples.size()[2:],
    )


def trace_helper(
    x: List[List[List[int]]], y: List[List[List[float]]], labels: List[str]
) -> Tuple[List[go.Scatter], List[str]]:
    """
    this function gets results prepared by a plot-related function and
    outputs a tuple including plotly object and its corresponding legend.
    """
    all_traces = []
    num_chains = len(x)
    num_indices = len(x[0])
    for index in range(num_indices):
        trace = []
        for chain in range(num_chains):
            trace.append(
                go.Scatter(
                    x=x[chain][index],
                    y=y[chain][index],
                    mode="lines",
                    name="chain" + str(chain),
                )
            )
        all_traces.append(trace)
    return (all_traces, labels)


def plot_helper(
    query_samples: Tensor, func: Callable
) -> Tuple[List[go.Scatter], List[str]]:
    """
    this function executes a plot-related function, passed as input parameter func, and
    outputs a tuple including plotly object and its corresponding legend.
    """
    num_chain, num_samples, single_sample_sz = _samples_info(query_samples)

    x_axis, y_axis, all_labels = [], [], []
    for chain in range(num_chain):
        flattened_data = query_samples[chain].reshape(num_samples, -1)
        numel = flattened_data[0].numel()
        x_axis_data, y_axis_data, labels = [], [], []
        for i in range(numel):
            index = np.unravel_index(i, single_sample_sz)
            data = flattened_data[:, i]
            partial_label = f" for {list(index)}"

            x_data, y_data = func(data.detach())
            x_axis_data.append(x_data)
            y_axis_data.append(y_data)
            labels.append(partial_label)
        x_axis.append(x_axis_data)
        y_axis.append(y_axis_data)
        all_labels.append(labels)
    return trace_helper(x_axis, y_axis, all_labels[0])


def autocorr(x: Tensor) -> Tuple[List[int], List[float]]:
    def autocorr_calculation(x: Tensor, lag: int) -> Tensor:
        y1 = x[: (len(x) - lag)]
        y2 = x[lag:]

        sum_product = (
            (y1 - (x.mean(dim=0).expand(y1.size())))
            * (y2 - (x.mean(dim=0).expand(y2.size())))
        ).sum(0)
        return sum_product / ((len(x) - lag) * torch.var(x, dim=0))

    max_lag = x.size(0)
    y_axis_data = [autocorr_calculation(x, lag).item() for lag in range(max_lag)]
    x_axis_data = list(range(max_lag))
    return (x_axis_data, y_axis_data)


def trace_plot(x: Tensor) -> Tuple[List[int], Tensor]:
    return (list(range(x.size(0))), x)
