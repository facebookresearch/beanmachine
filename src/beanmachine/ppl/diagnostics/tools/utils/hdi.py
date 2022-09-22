# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Union


def interval(
    data: List[float],
    hdi_probability: float,
) -> Dict[str, Union[List[float], float]]:
    """Find the lower and upper bound of the Highest Density Interval (HDI).

    Parameters
    ----------
    data : List[float]
        Raw random variable data from the model.
    hdi_probability : float
        The highest density interval probability to use when calculating the HDI.

    Returns
    -------
    Dict[str, Union[List[float], float]]
        A dictionary defining the lower and upper bound for the HDI of the given data,
        as well as the indices of the lower and upper bound for the sorted raw data.
    """
    N = len(data)
    sorted_data = sorted(data)
    stop_index = math.floor(hdi_probability * N)
    start_index = N - stop_index
    left_data = sorted_data[stop_index:]
    right_data = sorted_data[:start_index]
    hdi = []
    for i in range(len(left_data)):
        hdi.append(left_data[i] - right_data[i])
    lower_index = hdi.index(min(hdi))
    upper_index = lower_index + stop_index
    return {
        "lower_bound": sorted_data[lower_index],
        "upper_bound": sorted_data[upper_index],
        "lower_bound_index": lower_index,
        "upper_bound_index": upper_index,
    }


def data(
    rv_data: List[float],
    marginal_x: List[float],
    marginal_y: List[float],
    hdi_probability: float,
) -> Dict[str, Union[List[float], float]]:
    """Construct x and y arrays from the HDI bounds using the raw random variable dataself.

    It also returns the actual HDI bounds of the random variable data.

    Parameters
    ----------
    rv_data : List[float]
        Raw random variable data from the model.
    marginalX : List[float]
        The support of the Kernel Density Estimate of the random variable.
    marginalY : List[float]
        The Kernel Density Estimate of the random variable.
    hdiProbability : float
    The highest density interval probability to use when alculating the HDI.

    Returns
    -------
    Dict[str, Union[List[float], float]]
        The x array for the HDI data is the "lower" key, while the y array for the HDI
        data is the "upper" key and the "base" key is the support along the axis the HDI
        is calculated. This nomenclature mirrors the Bokeh Band object, which is the
        annotation type we will use when rendering the HDI intervals.
    """
    hdi = interval(rv_data, hdi_probability)
    base = []
    upper = []
    for i in range(len(marginal_x)):
        if marginal_x[i] <= hdi["upper_bound"] and marginal_x[i] >= hdi["lower_bound"]:
            base.append(marginal_x[i])
            upper.append(marginal_y[i])
    return {
        "base": base,
        "lower": [0] * len(base),
        "upper": upper,
        "lower_bound": hdi["lower_bound"],
        "upper_bound": hdi["upper_bound"],
    }
