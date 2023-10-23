# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import math
from collections import defaultdict

import torch


class BinSegChangepoint:
    """
    Offline changepoint detection using Binary Segmentation. This method
    greedily selects the changepoint that minimizes the negative log likelihood
    (NLL) to iteratively segment the time series.

    [1] Charles Truong, Laurent Oudre, Nicolas Vayatis, "Selective review of
    offline change point detection methods", https://arxiv.org/abs/1801.00718.

    :param int min_segment: the minimum length of a time series segment.
    :param float regularization: Optional regularization term for least squares
        to estimate the regression parameters.
    :param float penalty_weight: Overfitting penalty to add to the cost function
        when adding a changepoint. This is specified as a fraction w.r.t. the
        NLL of the original segment.
    :param int max_changepoints: Maximum number of changepoints to return. If
        the number of changepoints found is larger than `max_changepoints`, then
        `max_changepoints` number of changepoints with the lowest cost are
        returned.
    """

    def __init__(
        self,
        min_segment=10,
        regularization=0.0,
        penalty_weight=1e-2,
        max_changepoints=10,
    ):
        self.min_segment = min_segment
        self.regularization = regularization
        self.penalty_weight = penalty_weight
        self.max_changepoints = max_changepoints
        self._cost_cache = defaultdict(lambda: float("inf"))

    def nll(self, x, y):
        # MLE for penalized least squares
        theta = torch.inverse(x.T @ x + self.regularization * torch.eye(2)) @ (x.T @ y)
        n = len(y)
        const_term = -n / 2 * math.log(2 * math.pi)
        err = y - x @ theta
        param_term = -0.5 * torch.dot(err, err)
        return -(const_term + param_term)

    def get_cost(self, x, y, i, j):
        """
        The cost of a segment (i, j) is given by Negative Log Likelihood of the
        observations in (i, j), where the parameters of the regression are
        estimated by maximizing the likelihood.

        :param torch.Tensor x: 1-D torch tensor designating the independent
            axis.
        :param torch.Tensor y: 1-D torch tensor for the response variable over
            which to carry out changepoint detection.
        :param int i: lower limit (inclusive) for the changepoint search.
        :param int j: upper limit (exclusive) for the changepoint search.
        """
        if (i, j) in self._cost_cache:
            return self._cost_cache[(i, j)]
        x_seg = x[i:j]
        y_seg = y[i:j]
        cost = self.nll(x_seg, y_seg)
        self._cost_cache[(i, j)] = cost
        return cost

    def _changepoints_segment(self, x, y, i, j, penalty):
        orig_cost = self.get_cost(x, y, i, j)
        min_cost = orig_cost
        min_idx = None
        for k in range(i + self.min_segment, j - self.min_segment):
            cost_seg_1 = self.get_cost(x, y, i, k)
            cost_seg_2 = self.get_cost(x, y, k, j)
            total_cost = cost_seg_1 + cost_seg_2 + penalty
            if total_cost < min_cost:
                min_cost = total_cost
                min_idx = k
        if min_idx is None:
            return [], []
        seg_score = orig_cost - min_cost
        seg_lhs, score_lhs = self._changepoints_segment(x, y, i, min_idx + 1, penalty)
        seg_rhs, score_rhs = self._changepoints_segment(x, y, min_idx + 1, j, penalty)
        return seg_lhs + [min_idx] + seg_rhs, score_lhs + [seg_score] + score_rhs

    def get_changepoints(self, x, y):
        """
        The cost of adding a changepoint t to a segment (i, j) is given by:
            Cost(i, j, t) = NLL(i, t) + NLL(t, j) + penalty_weight * NLL(0, n)

        where NLL = Negative Log Likelihood of the observations, where the
        parameters of the regression are estimated by maximizing the likelihood.
        """
        n = len(y)
        assert len(x) == n, "`x` and `y` should have the same length."
        assert x.dim() == 1, "1-D tensor expected for `x`"
        assert y.dim() == 1, "1-D tensor expected for `y`"
        x = x.unsqueeze(-1)
        x = torch.cat([x, torch.ones_like(x)], -1)
        penalty = self.penalty_weight * self.get_cost(x, y, 0, n)
        changepoints, scores = self._changepoints_segment(x, y, 0, n, penalty)
        if len(changepoints) > self.max_changepoints:
            filtered = list(
                zip(
                    *(
                        sorted(zip(scores, changepoints), reverse=True)[
                            : self.max_changepoints
                        ]
                    )
                )
            )[1]
            changepoints = sorted(filtered)  # sort changepoints by location
        return changepoints
