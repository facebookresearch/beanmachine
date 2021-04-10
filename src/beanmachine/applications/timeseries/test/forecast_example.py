import argparse
import time

import gpytorch as gp
import numpy as np
import pandas as pd
import torch
from gpytorch.kernels import ScaleKernel, PeriodicKernel, RBFKernel
from sts.data import df_to_tensor
from sts.gp.model import TimeSeriesExactGPModel


def load_dataset(path):
    births = pd.read_csv(path)
    births["date"] = pd.to_datetime(births[["year", "month", "day"]])
    births["log_births"] = np.log(births["births"])
    df_convert_fns = {"date": lambda x: (x - x.iloc[0]).dt.days}

    data = df_to_tensor(births, normalize_cols=True, df_convert_fns=df_convert_fns)

    x_train = data[:6000, ["date"]]
    y_train = data[:6000, "log_births"]
    x_test = data[6000:, ["date"]]
    y_test = data[6000:, "log_births"]
    return x_train, y_train, x_test, y_test


class GPTS(gp.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x.tensor, train_y.tensor, likelihood)
        norm = train_x.transforms["date"].inv.scale
        self.mean_module = gp.means.ConstantMean()
        self.slow_moving_trend = ScaleKernel(RBFKernel())
        self.slow_moving_trend.base_kernel.lengthscale = 1000.0 / norm
        self.slow_periodic = ScaleKernel(PeriodicKernel())
        self.slow_periodic.base_kernel.period_length = 365.25 / norm
        self.covar_module = self.slow_moving_trend + self.slow_periodic

    def forward(self, x):
        mean = self.mean_module(x)
        cov = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean, cov)


def train_model_gpytorch(x_train, y_train, training_iter=100):
    likelihood = gp.likelihoods.GaussianLikelihood()
    model = GPTS(x_train, y_train, likelihood)
    x_train, y_train = x_train.tensor, y_train.tensor

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(
        set(model.parameters())
        - {
            model.slow_moving_trend.base_kernel.raw_lengthscale,
            model.slow_periodic.base_kernel.raw_period_length,
        },
        lr=0.2,
    )
    mll = gp.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(x_train)
        loss = -mll(output, y_train)
        loss.backward()
        if i % 10 == 0:
            print(
                "Iter %d/%d - Loss: %.3f"
                % (
                    i + 1,
                    training_iter,
                    loss.item(),
                )
            )
        optimizer.step()


def train_model_bmts(x_train, y_train, training_iter=100):
    likelihood = gp.likelihoods.GaussianLikelihood()
    model = TimeSeriesExactGPModel(x_train, y_train, likelihood)
    model.cov.add_seasonality(365.25, time_axis="date", fix_period=True)
    model.cov.add_trend(1000.0, time_axis="date", fix_lengthscale=True)

    train_apply = model.train_init(torch.optim.Adam(model.trainable_params, lr=0.1))

    for i in range(training_iter):
        loss = train_apply(x_train, y_train)
        if i % 10 == 0:
            print(
                "Iter %d/%d - Loss: %.3f"
                % (
                    i + 1,
                    training_iter,
                    loss,
                )
            )


def main(args):
    x_train, y_train, x_test, y_test = load_dataset(args.path)
    if args.model not in ("gpy", "bmts"):
        raise ValueError(f"Unsupported --model ({args.model}) argument provided.")
    train_fn = train_model_gpytorch if args.model == "gpy" else train_model_bmts
    print(f"Calling {train_fn}...")
    start = time.time()
    train_fn(x_train, y_train, training_iter=args.num_training_iter)
    print(f"Training complete in {time.time() - start:.2f} s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", default="gpy", type=str, help="Either bmts or gpy."
    )
    parser.add_argument("-p", "--path", type=str, help="Path to births dataset.")
    parser.add_argument(
        "-n",
        "--num-training-iter",
        default=100,
        type=int,
        help="Number of training iterations.",
    )
    main(parser.parse_args())
