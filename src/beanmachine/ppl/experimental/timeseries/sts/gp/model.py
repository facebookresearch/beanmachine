# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import warnings
from collections import OrderedDict
from typing import Callable, Set, Tuple, Union

import gpytorch as gp
import numpy as np
import torch
from gpytorch.kernels import LinearKernel, RBFKernel
from gpytorch.models import ExactGP, GP
from gpytorch.priors import LogNormalPrior
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from sts.changepoints import BinSegChangepoint
from sts.data import DataTensor, df_to_tensor
from sts.gp.cov import ChangePoint, Covariance, Trend
from sts.gp.mean import Mean
from torch import nn
from torch.distributions import transforms


class TimeSeriesGPModel(GP):
    @property
    def trainable_params(self) -> Set[nn.Parameter]:
        return set(self.parameters()) - self.cov.fixed_params


class ExactGPAdditiveModel(ExactGP, TimeSeriesGPModel):
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
        self.cov = Covariance(sample_input, sample_output)

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
        mll = gp.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        def apply_fn(x, y):
            if isinstance(x, DataTensor):
                x = x.tensor
            if isinstance(y, DataTensor):
                y = y.tensor
            optim.zero_grad()
            with warnings.catch_warnings():
                # ignore PyTorch warnings from GPyTorch
                warnings.simplefilter("ignore", UserWarning)
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


class AutomaticForecastingGP(ExactGPAdditiveModel):
    """
    Variant of time series forecasting model using GP as discussed in [1].
    K = PER1 + PER2 + LIN + RBF1 + RBF2 + SM + WN,
    where WN is already included in `GaussianLikelihood`

    :param pd.DataFrame df_train: Dataframe object for training the time series
        model.
    :param str time_var: The column corresponding to the time axis.
    :param str response_var: The column responding to the response variable
        we are looking to forecast.
    :param changepoint_locations: List of changepoint locations. For daily forecasts,
        these are simply the indices along the time axis where the trend changes
        rapidly.
    :param list linear_covariates: The mean of the GP is modeled as an affine
        function of the linear covariates specified.
    param list nonlinear_covariates: An ARD RBF kernel is placed over the
        specified covariates and added to the covariance kernel.
    :param dict priors: Priors for the length and output scale for different
        kernels. This updates the default priors dict in `self.priors`.
    :param bool detect_changepoints: Whether to enable automatic changepoint
        detection.
    :param dict changepoint_detection_params: Parameters for automatic changepoint
        detection algorithm (currently, binary segmentation).
    :param bool disable_linear_trend: Whether to disable linear trend in the model;
        defaults to False
    :param bool disable_sm_kernel: Whether to disable Spectral Mixture kernel
        component. Defaults to `True`.

    [1] Corani, G., Benavoli, A., Augusto, J. and Zaffalon, M., 2020.
    Automatic Forecasting using Gaussian Processes. arXiv preprint arXiv:2009.08102.
    """

    def __init__(
        self,
        df_train,
        time_var,
        response_var,
        changepoint_locations=(),
        linear_covariates=(),
        nonlinear_covariates=(),
        priors=None,
        detect_changepoints=False,
        changepoint_detection_params=None,
        disable_linear_trend=False,
        disable_sm_kernel=True,
    ):
        self.time_var = time_var
        if not is_datetime(df_train[self.time_var]):
            raise ValueError(f"{self.time_var} column must have datetime64 dtype.")
        self.time_scale = (df_train[time_var].max() - df_train[time_var].min()).days
        if self.time_scale < 15:
            raise ValueError(
                "Unsuitable class for short-term time series with less than 15 days."
            )
        self.response_var = response_var
        self.linear_covariates = list(linear_covariates)
        self.nonlinear_covariates = list(nonlinear_covariates)
        # Default priors on normalized scale
        self.priors = {
            "per_yearly_outputscale": LogNormalPrior(-3.0, 1.0),
            "per_yearly_lengthscale": LogNormalPrior(0.2, 1.0),
            "per_weekly_outputscale": LogNormalPrior(-3.0, 1.0),
            "per_weekly_lengthscale": LogNormalPrior(-5.9, 1.0),
            "lin_outputscale": LogNormalPrior(-3.0, 1.0),
            "rbf_long_outputscale": LogNormalPrior(-3.0, 1.0),
            "rbf_long_lengthscale": LogNormalPrior(
                np.log(self.time_scale / 365.25), 0.5
            ),
            "rbf_short_outputscale": LogNormalPrior(-3.0, 1.0),
            "rbf_short_lengthscale": LogNormalPrior(
                np.log(self.time_scale / 365.25) - np.log(100.0), 0.5
            ),
        }
        if priors:
            self.priors.update(priors)
        self._linear_trend_disabled = disable_linear_trend
        self._sm_kernel_disabled = disable_sm_kernel
        self.transforms = {}
        self.train_data = self.preprocess_data(df_train)
        x_train, y_train = (
            self.train_data[
                :, [time_var] + self.linear_covariates + self.nonlinear_covariates
            ],
            self.train_data[:, response_var],
        )
        if detect_changepoints:
            params = changepoint_detection_params
            if params is None:
                params = {}
            if changepoint_locations:
                warnings.warn(
                    "`changepoint_locations` will not be used. Set `detect_changepoints`"
                    " to False when manually specifying changepoint locations."
                )
            x, y = self.train_data[:, time_var], self.train_data[:, response_var]
            min_segment_length = min(len(x) // 5, 100)
            default_params = {
                "min_segment": min_segment_length,
                "penalty_weight": 1e-3,
            }
            default_params.update(params)
            binseg = BinSegChangepoint(**default_params)
            self.changepoint_locations = binseg.get_changepoints(x.tensor, y.tensor)
        else:
            self.changepoint_locations = sorted(changepoint_locations)
        likelihood = gp.likelihoods.GaussianLikelihood()
        super().__init__(x_train, y_train, likelihood)
        self._construct_model()

    def preprocess_data(self, df):
        df = df[
            [self.time_var, self.response_var]
            + self.linear_covariates
            + self.nonlinear_covariates
        ]
        df_convert_fns = {
            self.time_var: lambda x: (x - x.iloc[0]).dt.total_seconds() / (24 * 3600)
        }
        if not self.transforms:
            self.transforms = {}
            self.transforms[self.time_var] = transforms.AffineTransform(0.0, 365.25).inv
            std_norm_cols = (
                self.linear_covariates + self.nonlinear_covariates + [self.response_var]
            )
            for col in std_norm_cols:
                mean = df[col].mean()
                std = df[col].std()
                self.transforms[col] = transforms.AffineTransform(mean, std).inv
        return df_to_tensor(
            df, df_convert_fns=df_convert_fns, normalize_cols=self.transforms
        )

    def _construct_model(self):
        if self.time_scale >= 730:
            self.cov.add_seasonality(
                time_axis=self.time_var,
                period_length=365.25,
                fix_period=True,
                outputscale_prior=self.priors["per_yearly_outputscale"],
                lengthscale_prior=self.priors["per_yearly_lengthscale"],
                name="Yearly Seasonality",
            )
        self.cov.add_seasonality(
            time_axis=self.time_var,
            period_length=7,
            fix_period=True,
            outputscale_prior=self.priors["per_weekly_outputscale"],
            lengthscale_prior=self.priors["per_weekly_lengthscale"],
            name="Weekly Seasonality",
        )
        sample_input = self.cov.sample_input
        if not self.changepoint_locations:
            if not self._linear_trend_disabled:
                self.cov.add_trend(
                    time_axis=self.time_var,
                    kernel_cls=LinearKernel,
                    outputscale_prior=self.priors["lin_outputscale"],
                    name="Linear Trend",
                )
            self.cov.add_trend(
                time_axis=self.time_var,
                kernel_cls=RBFKernel,
                lengthscale=self.time_scale,
                outputscale_prior=self.priors["rbf_long_outputscale"],
                lengthscale_prior=self.priors["rbf_long_lengthscale"],
                lengthscale_constraint=gp.constraints.GreaterThan(
                    0.5 * self.time_scale / 365.25
                ),
                name="RBF Trend - Long",
            )
        else:
            changepoints = torch.as_tensor(self.changepoint_locations) / 365.25
            rbf_kernels = []
            lin_kernels = []
            time_scale = []
            periods = (
                [sample_input.min()]
                + list(self.changepoint_locations)
                + [sample_input.min() + self.time_scale]
            )
            for i in range(1, len(periods)):
                local_time_scale = periods[i] - periods[i - 1]
                time_scale.append(local_time_scale / 365.25)
                rbf_kernels.append(
                    Trend(
                        sample_input,
                        self.time_var,
                        kernel_cls=RBFKernel,
                        lengthscale=local_time_scale,
                        outputscale_prior=self.priors["rbf_long_outputscale"],
                        lengthscale_prior=LogNormalPrior(
                            np.log(2 * local_time_scale / 365.25), 0.5
                        ),
                        lengthscale_constraint=gp.constraints.GreaterThan(
                            0.5 * local_time_scale / 365.25
                        ),
                    )
                )
                if not self._linear_trend_disabled:
                    lin_kernels.append(
                        Trend(
                            sample_input,
                            self.time_var,
                            kernel_cls=LinearKernel,
                            outputscale_prior=self.priors["lin_outputscale"],
                        )
                    )
            steep = [
                1 / min(time_scale[i - 1], time_scale[i])
                for i in range(1, len(time_scale))
            ]
            self.cov.append(
                ChangePoint(
                    sample_input,
                    self.time_var,
                    rbf_kernels,
                    changepoint_location=changepoints,
                    changepoint_steep=steep,
                    fix_changepoint_location=True,
                    fix_changepoint_steep=True,
                    name="Changepoint - RBF",
                )
            )
            if lin_kernels:
                self.cov.append(
                    ChangePoint(
                        sample_input,
                        self.time_var,
                        lin_kernels,
                        changepoint_location=changepoints,
                        changepoint_steep=steep,
                        fix_changepoint_location=True,
                        fix_changepoint_steep=True,
                        name="Changepoint - Linear",
                    )
                )

        self.cov.add_trend(
            time_axis=self.time_var,
            kernel_cls=RBFKernel,
            lengthscale=1.0,
            outputscale_prior=self.priors["rbf_short_outputscale"],
            lengthscale_prior=self.priors["rbf_short_lengthscale"],
            name="RBF Trend - Short",
        )
        if self.linear_covariates:
            self.mean.add_regression(self.linear_covariates)
        if self.nonlinear_covariates:
            self.cov.add_regression(features=self.nonlinear_covariates)

        if not self._sm_kernel_disabled:
            self.cov.add_spectral_mixture(
                time_axis=self.time_var,
                num_mixtures=5,
                name="Spectral Mixture",
            )
