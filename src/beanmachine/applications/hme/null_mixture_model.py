# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
from typing import List

import beanmachine.graph as bmgraph
import numpy as np
import pandas as pd

from .abstract_linear_model import AbstractLinearModel
from .abstract_model import AbstractModel
from .configs import ModelConfig


logger = logging.getLogger("hme")


class NullMixtureMixedEffectModel(AbstractLinearModel):
    """
    Represents an generalized linear mixed effects model with optional null mixture.
    """

    def __init__(self, data: pd.DataFrame, model_config: ModelConfig) -> None:
        super().__init__(data, model_config)
        self.build_graph()

    def build_graph(self) -> None:
        self.fixed_effects, self.random_effects = AbstractModel.parse_formula(
            self.model_config.mean_regression.formula
        )
        if self.model_config.priors:
            self.prior_configs.update(self.model_config.priors)

        self.queries.clear()
        self.query_map.clear()

        self.g = bmgraph.Graph()
        self._set_priors()
        self._customize_priors()
        self._initialize_likelihood()
        for index, row in self.data.iterrows():
            fe_i = self._add_fixed_effects_byrow(row, self.fixed_effects_params)
            re_i = self._add_random_effects_byrow(row, (self.re_dist, self.re_value))
            if self.model_config.mean_mixture.use_asymmetric_modes:
                fe_neg_i = self._add_fixed_effects_byrow(
                    row, self.fixed_effects_params_neg
                )
                if self.model_config.mean_mixture.use_partial_asymmetric_modes:
                    re_neg_i = self._add_random_effects_byrow(
                        row, (self.re_dist, self.re_value_neg)
                    )
                else:
                    re_neg_i = self._add_random_effects_byrow(
                        row, (self.re_dist_neg, self.re_value_neg)
                    )
            else:
                fe_neg_i, re_neg_i = None, None
            if self.model_config.mean_mixture.use_bimodal_alternative:
                fere_i = self._add_bimodal_alternative(
                    [fe_i, re_i], [fe_neg_i, re_neg_i]
                )
            else:
                fere_i = self.g.add_operator(bmgraph.OperatorType.ADD, [fe_i, re_i])
            self._add_observation_byrow(index, row, fere_i)

    def _initialize_likelihood(self) -> None:
        self.sei = None
        # fixed and random effects component
        self.fixed_effects_params = self._initialize_fixed_effect_nodes()
        (
            self.re_dof,
            self.re_scale,
            self.re_dist,
            self.re_value,
        ) = self._initialize_random_effect_nodes(
            self.model_config.mean_regression.random_effect_distribution == "t"
        )
        self.queries.update(
            fixed_effect=self.fixed_effects_params,
            re_scale=self.re_scale,
            re_value=self.re_value,
        )
        if self.model_config.mean_regression.random_effect_distribution == "t":
            self.queries["re_dof"] = self.re_dof
        # mixture component
        if self.model_config.mean_mixture.use_null_mixture:
            prob_h_prior = (
                self.customized_priors["prob_h"]
                if "prob_h" in self.customized_priors
                else self.beta_prior
            )
            self.prob_h = self.g.add_operator(
                bmgraph.OperatorType.SAMPLE, [prob_h_prior]
            )
            self.h_dist = self.g.add_distribution(
                bmgraph.DistributionType.BERNOULLI,
                bmgraph.AtomicType.BOOLEAN,
                [self.prob_h],
            )
            self.h_all = {}
            self.mu_H1_all = {}
            self.queries.update(prob_h=self.prob_h, h=self.h_all, mu_H1=self.mu_H1_all)
            if self.model_config.mean_mixture.use_bimodal_alternative:
                prob_sign_prior = (
                    self.customized_priors["prob_sign"]
                    if "prob_sign" in self.customized_priors
                    else self.beta_prior
                )
                self.prob_sign = self.g.add_operator(
                    bmgraph.OperatorType.SAMPLE, [prob_sign_prior]
                )
                self.sign_dist = self.g.add_distribution(
                    bmgraph.DistributionType.BERNOULLI,
                    bmgraph.AtomicType.BOOLEAN,
                    [self.prob_sign],
                )
                self.queries["prob_sign"] = self.prob_sign
                if self.model_config.mean_mixture.use_asymmetric_modes:
                    # additional fixed and random effects
                    self.fixed_effects_params_neg = (
                        self._initialize_fixed_effect_nodes()
                    )
                    self.re_value_neg = {re: {} for re in self.random_effects}
                    self.queries.update(
                        fixed_effect_neg=self.fixed_effects_params_neg,
                        re_value_neg=self.re_value_neg,
                    )
                    if not self.model_config.mean_mixture.use_partial_asymmetric_modes:
                        (
                            self.re_dof_neg,
                            self.re_scale_neg,
                            self.re_dist_neg,
                            _,
                        ) = self._initialize_random_effect_nodes(
                            self.model_config.mean_regression.random_effect_distribution
                            == "t"
                        )
                        self.queries["re_scale_neg"] = self.re_scale_neg
                        if (
                            self.model_config.mean_regression.random_effect_distribution
                            == "t"
                        ):
                            self.queries["re_dof_neg"] = self.re_dof_neg
        else:
            self.yhat_all = {}
            self.queries["yhat"] = self.yhat_all

    def _add_bimodal_alternative(self, mu_parents: List, mu_neg_parents: List) -> int:
        log_mu = self.g.add_operator(bmgraph.OperatorType.ADD, mu_parents)
        exp_log_mu = self.g.add_operator(bmgraph.OperatorType.EXP, [log_mu])
        mu_pos = self.g.add_operator(bmgraph.OperatorType.TO_REAL, [exp_log_mu])
        if self.model_config.mean_mixture.use_asymmetric_modes:
            log_mu_neg = self.g.add_operator(bmgraph.OperatorType.ADD, mu_neg_parents)
            exp_log_mu_neg = self.g.add_operator(bmgraph.OperatorType.EXP, [log_mu_neg])
            real_exp_log_mu_neg = self.g.add_operator(
                bmgraph.OperatorType.TO_REAL, [exp_log_mu_neg]
            )
            mu_neg = self.g.add_operator(
                bmgraph.OperatorType.NEGATE, [real_exp_log_mu_neg]
            )
        else:
            mu_neg = self.g.add_operator(bmgraph.OperatorType.NEGATE, [mu_pos])
        sign_i = self.g.add_operator(bmgraph.OperatorType.SAMPLE, [self.sign_dist])
        fere_i = self.g.add_operator(
            bmgraph.OperatorType.IF_THEN_ELSE, [sign_i, mu_pos, mu_neg]
        )
        return fere_i

    def _add_observation_byrow(self, index: int, row: pd.Series, fere_i: int) -> None:
        if self.model_config.mean_mixture.use_null_mixture:
            hi = self.g.add_operator(bmgraph.OperatorType.SAMPLE, [self.h_dist])
            self.h_all[index] = hi
            yhat = self.g.add_operator(
                bmgraph.OperatorType.IF_THEN_ELSE, [hi, fere_i, self.zero]
            )
            self.mu_H1_all[index] = fere_i
        else:
            yhat = fere_i
            self.yhat_all[index] = yhat
        if self.model_config.mean_regression.stderr:
            sei = self.g.add_constant_pos_real(
                row[self.model_config.mean_regression.stderr]
            )
        elif self.sei:
            sei = self.sei
        else:
            sei = self.g.add_operator(
                bmgraph.OperatorType.SAMPLE, [self.halfcauchy_prior]
            )
            self.sei = sei
            self.queries["se"] = sei

        y_dist = None
        if self.model_config.mean_regression.distribution == "bernoulli":
            assert self.model_config.mean_regression.link == "logit"
            y_dist = self.g.add_distribution(
                bmgraph.DistributionType.BERNOULLI_LOGIT,
                bmgraph.AtomicType.BOOLEAN,
                [yhat],
            )
        elif self.model_config.mean_regression.distribution == "normal":
            assert self.model_config.mean_regression.link == "identity"
            y_dist = self.g.add_distribution(
                bmgraph.DistributionType.NORMAL, bmgraph.AtomicType.REAL, [yhat, sei]
            )
        else:
            raise RuntimeError(
                "{d} distribution with {l} link function is not supported".format(
                    d=self.model_config.mean_regression.distribution,
                    l=self.model_config.mean_regression.link,
                )
            )
        yi = self.g.add_operator(bmgraph.OperatorType.SAMPLE, [y_dist])
        self.g.observe(yi, row[self.model_config.mean_regression.outcome])

    def predict(
        self, new_data: pd.DataFrame, post_samples: pd.DataFrame
    ) -> pd.DataFrame:
        # FIXME: update the method to support customized prediction goal
        pred_df = pd.DataFrame()
        for _, row in new_data.iterrows():
            pred_row = self._predict_fere_byrow(row, post_samples)
            if self.model_config.mean_mixture.use_bimodal_alternative:
                pred_row = np.exp(pred_row)
            pred_df = pred_df.append(pred_row, ignore_index=True)
        return pred_df
