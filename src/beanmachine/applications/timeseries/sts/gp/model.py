from collections import OrderedDict
from typing import Callable, Set, Tuple, Union

import gpytorch as gp
import torch
from gpytorch.models import GP, ExactGP
from sts.data import DataTensor
from sts.gp.cov import Covariance
from sts.gp.mean import Mean
from torch import nn


class TimeSeriesGPModel(GP):
    @property
    def trainable_params(self) -> Set[nn.Parameter]:
        return set(self.parameters()) - self.cov.fixed_params


class TimeSeriesExactGPModel(ExactGP, TimeSeriesGPModel):
    """
    Univariate time series model using exact GP.

    :param sample_input: a sample :class:`DataTensor` object that represents
        the inputs of interest.
    :param sample_output: a sample :class:`DataTensor` object that represents
        the output from the time series model.
    :param likelihood: a gpytorch :class:`~gp.likelihoods.Likelihood` object
        that specifies the mapping from `f(X)` to observed data.
    """

    def __init__(
        self,
        sample_input: DataTensor,
        sample_output: DataTensor,
        likelihood: gp.likelihoods.Likelihood,
    ):
        super().__init__(sample_input.tensor, sample_output.tensor, likelihood)
        self.mean = Mean(sample_input)
        self.cov = Covariance(sample_input)

    def forward(self, x: Union[torch.Tensor, DataTensor]):
        if isinstance(x, DataTensor):
            x = x.tensor
        mean_x = self.mean(x).squeeze(-1)
        covar_x = self.cov(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, *args, **kwargs):
        args = [x.tensor if isinstance(x, DataTensor) else x for x in args]
        return super().__call__(*args, **kwargs)

    def train_init(self, optim: torch.optim.Optimizer) -> Callable:
        """
        Return a callable that can be used to run training iterations for the
        time series model.

        :param optim: the :class:`~torch.optim.Optimizer` to use for minimizing
            the loss function.
        :return: Returns a callable which runs a single iteration for the
            optimizer when called with the training inputs and outputs and
            returns the loss.
        """
        self.train()
        self.requires_grad_(True)
        mll = gp.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        def apply_fn(x, y):
            if isinstance(x, DataTensor):
                x = x.tensor
            if isinstance(y, DataTensor):
                y = y.tensor
            optim.zero_grad()
            loss = -mll(self(x), y)
            loss.backward()
            optim.step()
            return loss.item()

        return apply_fn

    def predict(
        self, x: Union[torch.Tensor, DataTensor]
    ) -> gp.distributions.MultivariateNormal:
        """
        Run the model to generate predictions for the given input tensor.

        :param x: input data.
        :return: output tensor that represents the prediction from the model.
        """
        if isinstance(x, DataTensor):
            x = x.tensor
        self.eval()
        self.requires_grad_(False)
        with torch.no_grad(), gp.settings.fast_pred_var():
            return self.likelihood(self(x))

    def _component_cov(self, test_X, k):
        # See the note on additive decomposition: https://www.cs.toronto.edu/~duvenaud/cookbook/
        train_X = self.train_inputs[0]
        test_train_covar = k(test_X, train_X)
        test_test_covar = k(test_X, test_X)
        pred_strategy = self.prediction_strategy
        train_train_covar = pred_strategy.lik_train_train_covar
        train_train_covar_inv_root = train_train_covar.root_inv_decomposition().root
        covar_inv_quad_form_root = test_train_covar.matmul(train_train_covar_inv_root)
        kxx_inv = covar_inv_quad_form_root @ covar_inv_quad_form_root.transpose(-1, -2)
        return test_test_covar - kxx_inv

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
        self.requires_grad_(False)
        train_X = self.train_inputs[0]
        with torch.no_grad(), gp.settings.fast_pred_var():
            alpha = self.prediction_strategy.mean_cache
            mean_parts = OrderedDict(
                [(name, p(x)) for name, p in self.mean.parts.items()]
            )
            cov_parts = OrderedDict()
            for name, p in self.cov.parts.items():
                mean = p(x, train_X) @ alpha
                cov = self._component_cov(x, p.kernel)
                cov_parts[name] = gp.distributions.MultivariateNormal(mean, cov)
        return mean_parts, cov_parts
