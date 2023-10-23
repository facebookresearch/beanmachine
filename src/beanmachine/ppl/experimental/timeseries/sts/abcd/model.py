# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import OrderedDict, Tuple, Union

import gpytorch as gp
import torch
from sts.abcd.expression import KernelExpression
from sts.abcd.utils import is_base_kernel
from sts.data import DataTensor
from sts.gp.mean import Mean
from sts.gp.model import ExactGPAdditiveModel
from sts.metrics import AIC, BIC


class ABCDExactGPModel(ExactGPAdditiveModel):
    """
    Univariate time series model for Automatic Bayesian Covariance Discovery (ABCD) [1].
    Reference:
    [1] Lloyd, James, et al. "Automatic construction and natural-language description of
    nonparametric regression models." Proceedings of the AAAI Conference on Artificial
    Intelligence. Vol. 28. No. 1. 2014.

    :param train_inputs: a sample :class:`DataTensor` object that represents
        the inputs of interest.
    :param train_targets: a sample :class:`DataTensor` object that represents
        the output from the time series model.
    :param likelihood: a gpytorch :class:`~gp.likelihoods.Likelihood` object
        that specifies the mapping from `f(X)` to observed data.
    :param kernel: a gpytorch :class:`~gp.kernels.Kernel` object that specifies
        the candidate kernel to optimize parameters and evaluate with BIC score.
    """

    def __init__(
        self,
        train_inputs: DataTensor,
        train_targets: DataTensor,
        likelihood: gp.likelihoods.Likelihood,
        kernel: gp.kernels.Kernel,
    ):
        super().__init__(train_inputs, train_targets, likelihood)
        self.mean = Mean(train_inputs)
        self.cov = kernel

    def optimize_params(self, learning_rate=0.01, num_epochs=1000):
        """
        Train the model to optimize parameters.
        :param learning_rate: the learning rate of the model.
        :param num_epochs: the number of epochs to train the model.
        """
        trainer = self.train_init(torch.optim.Adam(self.parameters(), lr=learning_rate))

        for _ in range(num_epochs):
            self.loss = trainer(self.train_inputs[0], self.train_targets)

    def score(self, method="BIC") -> torch.Tensor:
        """
        Score the model with method.
        :param method: the scoring method, can be BIC, AIC, NLL.
        :return: the score.
        """
        with torch.no_grad():
            n = self.train_inputs[0].shape[0]
            nll = self.loss * n
            if method == "BIC":
                return BIC(
                    nll,
                    len(list(self.cov.parameters())),
                    n,
                )
            elif method == "AIC":
                return AIC(
                    nll,
                    len(list(self.cov.parameters())),
                )
            elif method == "NLL":
                return nll

    def decompose_timeseries(
        self, x: Union[torch.Tensor, DataTensor]
    ) -> Tuple[OrderedDict, OrderedDict]:
        """
        Run the model to generate predictions for the given input tensor, and
        return the decomposition over the mean and additive covariance kernels.

        :param x: input data.
        :return: tuple of mean and covariance decompositions, which are dicts
            keyed by component names.
        """
        self.eval()
        train_X = self.train_inputs[0]
        with torch.no_grad(), gp.settings.fast_pred_var():
            alpha = self.prediction_strategy.mean_cache
            mean_parts = OrderedDict(
                [(name, p(x)) for name, p in self.mean.parts.items()]
            )
            cov_parts = OrderedDict()
            kern_exp = KernelExpression(self.cov)
            kernel_additive = kern_exp.additive_form_kernel()
            if is_base_kernel(kernel_additive):
                kernel_list = [kernel_additive]
            else:
                kernel_list = list(kernel_additive.kernels)
            for p in kernel_list:
                mean = p(x, train_X) @ alpha
                cov = self._component_cov(x, p)
                name = repr(KernelExpression(p))
                suffix = 1
                while name in cov_parts.keys():
                    name = name + str(suffix)
                    suffix += 1
                cov_parts[name] = gp.distributions.MultivariateNormal(mean, cov)
        return mean_parts, cov_parts
