import unittest
from typing import Dict

import beanmachine.ppl.diagnostics.common_statistics as common_statistics
import pandas as pd
import torch
import torch.distributions as dist
from beanmachine.ppl.diagnostics.diagnostics import Diagnostics
from beanmachine.ppl.inference.single_site_ancestral_mh import (
    SingleSiteAncestralMetropolisHastings,
)
from beanmachine.ppl.model.statistical_model import sample


diri_dis = dist.Dirichlet(
    torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [2.0, 3.0, 1.0]])
)

beta_dis = dist.Beta(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([9.0, 8.0, 7.0]))

normal_dis = dist.Normal(torch.tensor([0.0, 1.0, 2.0]), torch.tensor([0.5, 1.0, 1.5]))


@sample
def diri(i, j):
    return diri_dis


@sample
def beta(i):
    return beta_dis


@sample
def normal():
    return normal_dis


@sample
def foo():
    return dist.Normal(0, 1)


def dist_summary_stats() -> Dict[str, torch.tensor]:
    exact_mean = {
        "beta": beta_dis.mean.reshape(-1),
        "diri": diri_dis.mean.reshape(-1),
        "normal": normal_dis.mean.reshape(-1),
    }

    exact_std = {
        "beta": torch.sqrt(beta_dis.variance.reshape(-1)),
        "diri": torch.sqrt(diri_dis.variance.reshape(-1)),
        "normal": torch.sqrt(normal_dis.variance.reshape(-1)),
    }
    exact_CI_2_5 = {"normal": normal_dis.mean - 1.96 * torch.sqrt(normal_dis.variance)}
    exact_CI_50 = {"normal": normal_dis.mean}
    exact_CI_97_5 = {"normal": normal_dis.mean + 1.96 * torch.sqrt(normal_dis.variance)}

    exact_stats = {
        "avg": exact_mean,
        "std": exact_std,
        "2.5%": exact_CI_2_5,
        "50%": exact_CI_50,
        "97.5%": exact_CI_97_5,
    }

    return exact_stats


class DiagnosticsTest(unittest.TestCase):
    def test_basic_diagnostics(self) -> pd.DataFrame:
        def _inference_evaulation(summary: pd.DataFrame):
            exact_stats = dist_summary_stats()

            for col in summary.columns:
                if not (col in exact_stats):
                    continue
                for dis, res in exact_stats[col].items():
                    query_res = summary.loc[summary.index.str.contains(f"^{dis}")]
                    for i, val in enumerate(query_res[col].values):
                        self.assertAlmostEqual(
                            val,
                            res[i].item(),
                            msg=f"query {query_res.index[i]} for {col}",
                            delta=0.2,
                        )

        mh = SingleSiteAncestralMetropolisHastings()
        samples = mh.infer([beta(0), diri(1, 5), normal()], {}, 5000, 2)

        out_df = Diagnostics(samples).summary()
        _inference_evaulation(out_df)

        out_df = Diagnostics(samples).summary([diri(1, 5), beta(0)])
        _inference_evaulation(out_df)

        out_df = Diagnostics(samples).summary(query_list=[diri(1, 5)], chain=1)
        _inference_evaulation(out_df)

        self.assertRaises(ValueError, Diagnostics(samples).summary, [diri(1, 3)])
        self.assertRaises(ValueError, Diagnostics(samples).summary, [diri(1, 5), foo()])

    def test_r_hat_column(self):
        mh = SingleSiteAncestralMetropolisHastings()
        samples = mh.infer([normal()], {}, 100, 2)
        diagnostics = Diagnostics(samples)

        out_df = diagnostics.summary()
        self.assertTrue("r_hat" in out_df.columns)

        out_df = diagnostics.summary(chain=0)
        self.assertTrue("r_hat" not in out_df.columns)

    def test_r_hat_no_column(self):
        mh = SingleSiteAncestralMetropolisHastings()
        samples = mh.infer([normal()], {}, 100, 1)
        out_df = Diagnostics(samples).summary()
        self.assertTrue("r_hat" not in out_df.columns)

    def test_r_hat(self):
        samples = torch.tensor([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])
        self.assertAlmostEqual(common_statistics.r_hat(samples), 2.3558, delta=0.001)
        self.assertAlmostEqual(
            common_statistics.split_r_hat(samples), 3.7193, delta=0.001
        )

    def test_r_hat_additional_dimension(self):
        samples = torch.tensor(
            [
                [[0.0, 2.0], [2.0, 4.0], [4.0, 8.0], [6.0, 0.0]],
                [[8.0, 12.0], [10.0, 6.0], [12.0, 1.0], [14.0, 2.0]],
                [[16.0, -5.0], [18.0, 4.0], [20.0, 2.0], [22.0, 4.0]],
            ]
        )
        dim1, dim2 = common_statistics.r_hat(samples)
        self.assertAlmostEqual(dim1, 3.2171, delta=0.001)
        self.assertAlmostEqual(dim2, 0.9849, delta=0.001)
        dim1, dim2 = common_statistics.split_r_hat(samples)
        self.assertAlmostEqual(dim1, 5.3385, delta=0.001)
        self.assertAlmostEqual(dim2, 1.0687, delta=0.001)

    def test_r_hat_error(self):
        samples = torch.tensor([0.0, 1.0, 2.0, 3.0])
        self.assertRaises(ValueError, common_statistics.r_hat, samples)
        self.assertRaises(ValueError, common_statistics.split_r_hat, samples)

    def test_effective_sample_size(self):
        samples = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
        n_eff = common_statistics.effective_sample_size(samples)
        self.assertAlmostEqual(n_eff, 2.6114, delta=0.001)

    def test_effective_sample_size_additional_dimension(self):
        samples = torch.tensor(
            [
                [[0.0, 2.0], [2.0, 4.0], [4.0, 8.0], [6.0, 0.0]],
                [[8.0, 12.0], [10.0, 6.0], [12.0, 1.0], [14.0, 2.0]],
                [[16.0, -5.0], [18.0, 4.0], [20.0, 2.0], [22.0, 4.0]],
            ]
        )
        dim1, dim2 = common_statistics.effective_sample_size(samples)
        self.assertAlmostEqual(dim1, 1.9605, delta=0.001)
        self.assertAlmostEqual(dim2, 15.1438, delta=0.001)

    def test_effective_sample_size_columns(self):
        mh = SingleSiteAncestralMetropolisHastings()
        samples = mh.infer([normal()], {}, 100, 2)
        out_df = Diagnostics(samples).summary()
        self.assertTrue("n_eff" in out_df.columns)
