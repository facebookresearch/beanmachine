from collections import OrderedDict
from typing import Iterable

import gpytorch as gp
import torch
from gpytorch.means import ConstantMean, LinearMean
from sts.data import DataTensor
from torch import nn


class RegressionMean(LinearMean):
    def __init__(self, sample_df: DataTensor, features: Iterable[str]):
        super().__init__(len(features), bias=False)
        self.active_dims = sample_df.get_index(features)

    def forward(self, x):
        x = x[:, self.active_dims]
        return super().forward(x)


class Mean(gp.means.Mean):
    """
    Represents the mean prediction from the model.

    :param sample_input: a sample :class:`DataTensor` object that represents
        the inputs of interest.
    """

    def __init__(self, sample_input: DataTensor):
        super().__init__()
        self.sample_input = sample_input
        self.parts = OrderedDict()
        self.means = nn.ModuleList()
        self.append(ConstantMean())

    def append(self, component: gp.means.Mean):
        """
        Append a :class:`gpytorch.means.Mean` module to the current model.

        :param component: [description]
        :type component: [type]
        """
        name = component._get_name()
        i = 1
        while name in self.parts:
            name = name + f"_{i}"
            i += 1
        component.name = name
        self.parts[name] = component
        self.means.append(component)

    def add_regression(self, features: Iterable[str]):
        """
        Add a linear regression component to the mean function of the GP.

        :param features: list of header names to be included as regressors.
        """
        if len(features) == 0:
            return
        self.append(RegressionMean(self.sample_input, features))

    def forward(self, input: torch.Tensor):
        return sum(c.forward(input) for c in self.means)
