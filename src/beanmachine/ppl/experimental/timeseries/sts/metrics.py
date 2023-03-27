# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
This implementation of CRPS metric follows the implementation in
https://github.com/TheClimateCorporation/properscoring.

The original copyright notice:
Copyright 2015 The Climate Corporation
Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""


import contextlib
import math
import warnings

import torch
import torch.distributions as dist


def mape(y_pred, y_true) -> torch.Tensor:
    """
    Mean Absolute Percentage Error for the predictions `y_pred` when compared
    to the baseline given by `y_true`.

    :param y_pred: predictions from the model.
    :param y_true: true observations.
    :return: torch scalar denoting the MAPE.
    """
    return torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100


def MAE(y_pred, y_true):
    """
    Mean Absolute Error for the predictions `y_pred` when compared
    to the baseline given by `y_true`.

    :param y_pred: predictions from the model.
    :param y_true: true observations.
    :return: torch scalar denoting the MAE.
    """
    return (y_pred - y_true).abs().mean()


def log_likelihood_error(y_pred, y_true, var):
    """
    Log_likelihood for the predictions `y_pred` when compared
    to the baseline given by `y_true` with variance `var`.
    This assumes a gaussian likelihood.

    :param y_pred: predictions from the model.
    :param y_true: true observations.
    :param var: variance from the model.
    :return: torch scalar denoting the log-likelihood.
    """
    return (
        -0.5 * torch.log(torch.tensor(2 * math.pi * var)).mean()
        - 0.5 * (((y_pred - y_true) ** 2) / var).mean()
    )


def BIC(nll: float, M: int, n: int):
    """
    BIC(M) = −2 log p(D | M ) + | M | log n
    |M| is the number of kernel parameters, p(D|M) is the marginal likelihood of the data, D,
    and n is the number of data points.
    :param nll: the negative marginal likelihood of the data, D
    :param M: the number of kernel parameters
    :param n: the number of data points
    :return: the BIC score.
    """
    return 2 * nll + M * torch.log(torch.tensor(n))


def AIC(nll: float, M: int):
    """
    AIC(M) = −2 log p(D | M ) + 2 | M |
    |M| is the number of kernel parameters, p(D|M) is the marginal likelihood of the data, D.
    :param nll: the negative marginal likelihood of the data, D
    :param M: the number of kernel parameters
    :return: the AIC score.
    """
    return 2 * nll + 2 * M


def move_axis_to_end(array, axis):
    return torch.moveaxis(array, axis, array.ndim - 1)


@contextlib.contextmanager
def suppress_warnings(msg=None):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", msg)
        yield


def crps_gaussian(x, mu, sigma):
    """
    Computes the CRPS of observations x relative to normally distributed
    forecasts with mean, mu, and standard deviation, sig.
    CRPS(N(mu, sig^2); x)
    Formula taken from Equation (5):
    Calibrated Probablistic Forecasting Using Ensemble Model Output
    Statistics and Minimum CRPS Estimation. Gneiting, Raftery,
    Westveld, Goldman. Monthly Weather Review 2004
    http://journals.ametsoc.org/doi/pdf/10.1175/MWR2904.1

    :param x : scalar or array_like
        The observation or set of observations.
    :param mu : scalar or array_like
        The mean of the forecast normal distribution
    :param sigma : scalar or array_like
        The standard deviation of the forecast distribution
    :return: torch.Tensor
        The CRPS of each observation x relative to mu and sig.
        The shape of the output array is determined by numpy
        broadcasting rules.
    """

    x = torch.as_tensor(x)
    mu = torch.as_tensor(mu)
    sigma = torch.as_tensor(sigma)
    # standardized x
    sx = (x - mu) / sigma
    pdf = dist.Normal(0.0, 1.0).log_prob(sx).exp()
    cdf = dist.Normal(0.0, 1.0).cdf(sx)
    pi_inv = 1.0 / torch.sqrt(torch.tensor(math.pi))
    crps = sigma * (sx * (2 * cdf - 1) + 2 * pdf - pi_inv)
    return crps


def _nanmean(v, *args, inplace=False, **kwargs):
    """
    Computes the mean of v excluding nan values. Similar to numpy.nanmean.
    :param v: torch.Tensor
        the tensor to compute its mean
    :param *args: list arguments to pass to sum
    :param **kwargs: key word arguments to pass to sum
    :return: torch.Tensor
        mean of v excluding nan.
    """
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (torch.logical_not(is_nan)).float().sum(
        *args, **kwargs
    )


def _crps_ensemble_vectorized(observations, forecasts, weights):
    """
    A simple but expensive O(n^2) implementation of CRPS for testing purposes
    This implementation is based on the identity:
    .. math::
        CRPS(F, x) = E_F|X - x| - 1/2 * E_F|X - X'|
    where X and X' denote independent random variables drawn from the forecast
    distribution F, and E_F denotes the expectation value under F.
    Hence it has runtime O(n^2) instead of O(n log(n)) where n is the number of
    ensemble members.
    """

    if weights.ndim > 0:
        weights = torch.where(
            torch.logical_not(torch.isnan(forecasts)), weights.double(), float("nan")
        )
        weights = weights / _nanmean(weights, axis=-1, keepdims=True)

    if observations.ndim == forecasts.ndim - 1:
        # sum over the last axis
        assert observations.shape == forecasts.shape[:-1]
        observations = observations.unsqueeze(-1)
        with suppress_warnings("Mean of empty slice"):
            score = _nanmean(weights * abs(forecasts - observations), -1)
        # insert new axes along last and second to last forecast dimensions so
        # forecasts_diff expands with broadcasting
        forecasts_diff = forecasts.unsqueeze(-1) - forecasts.unsqueeze(-2)
        weights_matrix = weights.unsqueeze(-1) * weights.unsqueeze(-2)
        with suppress_warnings("Mean of empty slice"):
            score += -0.5 * _nanmean(
                weights_matrix * abs(forecasts_diff), axis=(-2, -1)
            )
        return score
    elif observations.ndim == forecasts.ndim:
        # there is no 'realization' axis to sum over (this is a deterministic
        # forecast)
        return abs(observations - forecasts)


def crps_ensemble(observations, forecasts, issorted=False, axis=-1):
    """
    Calculate the continuous ranked probability score (CRPS) for a set of
    explicit forecast realizations.

    :param observations : float or array_like
        Observations float or array. Missing values (NaN) are given scores of
        NaN.
    :param forecasts : float or array_like
        Array of forecasts ensemble members, of the same shape as observations
        except for the axis along which CRPS is calculated (which should be the
        axis corresponding to the ensemble). If forecasts has the same shape as
        observations, the forecasts are treated as deterministic. Missing
        values (NaN) are ignored.
    :param issorted : bool, optional
        Optimization flag to indicate that the elements of `ensemble` are
        already sorted along `axis`.
    :param axis : int, optional
        Axis in forecasts and weights which corresponds to different ensemble
        members, along which to calculate CRPS.
    : return: Tensor
        CRPS for each ensemble forecast against the observations.
    """
    # TODO: Implement O(nlogn) version of the metric.
    warnings.warn(
        "This is a naive implementation that takes O(n^2) runtime, "
        "and is not suited for large number of observations."
    )
    observations = torch.as_tensor(observations)
    forecasts = torch.as_tensor(forecasts)
    if axis != -1:
        forecasts = move_axis_to_end(forecasts, axis)

    if not issorted:
        forecasts = torch.sort(forecasts, axis=-1).values

    weights = torch.ones_like(forecasts)

    return _crps_ensemble_vectorized(observations, forecasts, weights)
