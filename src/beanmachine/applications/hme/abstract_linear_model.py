# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import re as regx
from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple

import beanmachine.graph as bmgraph
import pandas as pd

from .abstract_model import AbstractModel
from .configs import (
    ModelConfig,
    PriorConfig,
    StructuredPriorConfig,
)


logger = logging.getLogger("hme")


class AbstractLinearModel(AbstractModel, metaclass=ABCMeta):
    """An abstract class for creating linear mixed effects model using BMGraph.

    :param data: observed training data
    :param model_config: model configuration parameters
    """

    def __init__(self, data: pd.DataFrame, model_config: ModelConfig) -> None:
        super().__init__(data, model_config)
        self.fixed_effects = []
        self.random_effects = []
        self.customized_priors = {}
        self.default_priors = {}

    @abstractmethod
    def _add_observation_byrow(self, index: int, row: pd.Series, fere_i: int) -> None:
        """
        This method defines the conditional distribution of the observation node
        given the linear component node: fere_i
        """

        pass

    def _set_priors(self) -> None:
        """Pre-defines the following common prior distributions:

        * Beta(alpha=1, beta=1),
        * Gamma(alpha=1, beta=1),
        * Half-Cauchy(gamma=1),
        * Half-Normal(sigma=1),
        * Normal(mu=0, sigma=2),
        * Non-standardized Student's t(df=3, mu=0, sigma=3).
        """

        self.zero = self.g.add_constant(0.0)
        self.one = self.g.add_constant_pos_real(1.0)
        self.two = self.g.add_constant_pos_real(2.0)
        self.three = self.g.add_constant_pos_real(3.0)

        self.beta_prior = self.g.add_distribution(
            bmgraph.DistributionType.BETA,
            bmgraph.AtomicType.PROBABILITY,
            [self.one, self.one],
        )
        self.gamma_prior = self.g.add_distribution(
            bmgraph.DistributionType.GAMMA,
            bmgraph.AtomicType.POS_REAL,
            [self.one, self.one],
        )
        self.halfcauchy_prior = self.g.add_distribution(
            bmgraph.DistributionType.HALF_CAUCHY,
            bmgraph.AtomicType.POS_REAL,
            [self.one],
        )
        self.halfnormal_prior = self.g.add_distribution(
            bmgraph.DistributionType.HALF_NORMAL,
            bmgraph.AtomicType.POS_REAL,
            [self.one],
        )
        self.normal_prior = self.g.add_distribution(
            bmgraph.DistributionType.NORMAL,
            bmgraph.AtomicType.REAL,
            [self.zero, self.two],
        )
        self.t_prior = self.g.add_distribution(
            bmgraph.DistributionType.STUDENT_T,
            bmgraph.AtomicType.REAL,
            [self.three, self.zero, self.three],
        )

    def _set_default_priors(self) -> None:
        self.default_priors = {}

        self.default_priors["fixed_effects"] = self.normal_prior
        self.default_priors["prob_h"] = self.beta_prior
        self.default_priors["prob_sign"] = self.beta_prior

    def _initialize_fixed_effect_nodes(self) -> Dict[str, int]:
        """Initializes fixed effect nodes in the graph, whose values are sampled from
        pre-specified prior distributions, defaults to Normal(mu=0, sigma=2).

        :return: a mapping of fixed effects to their corresponding BMGraph sample nodes
        """

        fixed_effects_params = {}

        for fe in self.fixed_effects:
            for key, val in self.customized_priors.items():
                pattern = "^" + key + r"(\[T.\w+\])*"
                if regx.match(pattern, fe):
                    fixed_effects_params[fe] = self.g.add_operator(
                        bmgraph.OperatorType.SAMPLE, [val]
                    )
                    break
            else:
                fixed_effects_params[fe] = self.g.add_operator(
                    bmgraph.OperatorType.SAMPLE,
                    [self.default_priors["fixed_effects"]],
                )

        return fixed_effects_params

    def _initialize_random_effect_nodes(self) -> Tuple[dict, dict, dict]:
        """Initializes random effect nodes in the graph. This includes assigning priors to
        the random effect nodes as well as assigning priors to hyper-parameters. The method
        allows for any flexible prior on random effects if the user desires to better model
        heavy tailed effects.

        :return: a tuple of dictionaries, which map random effects to their parameters, distribution, and sampled value nodes
        """

        re_dist, re_param = {}, {}

        for re in self.random_effects:
            if re in self.customized_priors:
                if isinstance(self.customized_priors[re], PriorConfig):
                    re_dist[re], re_param[re] = self._parse_re_prior_config(
                        self.customized_priors[re], re
                    )
                elif isinstance(self.customized_priors[re], StructuredPriorConfig):
                    if self.model_config.mean_mixture.use_bimodal_alternative:
                        raise ValueError(
                            "Structured Priors are not supported for bimodal alternatives!"
                        )
                    re_dist[re], re_param[re] = self._parse_re_structured_prior_config(
                        self.customized_priors[re], re
                    )
            # default
            else:
                re_scale = self.g.add_operator(
                    bmgraph.OperatorType.SAMPLE, [self.halfnormal_prior]
                )
                re_dist[re] = self.g.add_distribution(
                    bmgraph.DistributionType.NORMAL,
                    bmgraph.AtomicType.REAL,
                    [self.zero, re_scale],
                )
                re_param[re] = {"scale": re_scale}

        re_value = {re: {} for re in self.random_effects}

        return re_param, re_dist, re_value

    def _add_fixed_effects_byrow(self, row: pd.Series, params: Dict[str, int]) -> int:
        """Forms the systematic component from the fixed effects per subject (i.e. per row in the training data).
        In the other words, this method returns XB for fixed effects for a given obs (row) from the training data.

        :param row: one row of training data with fixed effect covariates
        :param params: a mapping of fixed effects to their corresponding nodes
        :return: bmgraph node that sums over all fixed effects for a given observation (i.e. a row from the training data)
        """

        if not self.fixed_effects:
            return self.zero
        fe_list = []
        for fe in self.fixed_effects:
            x = self.g.add_constant(row[fe])  # Q: what if x is pos_real or prob?
            x_param = self.g.add_operator(
                bmgraph.OperatorType.MULTIPLY, [x, params[fe]]
            )
            fe_list.append(x_param)
        if len(fe_list) < 2:
            return fe_list[0]
        return self.g.add_operator(bmgraph.OperatorType.ADD, fe_list)

    def _add_random_effects_byrow(self, row: pd.Series, params: Tuple) -> int:
        """Forms the systematic component from the random effects per subject (i.e. per row in the training data).
        In the other words, this method returns XZ for random effects for a given obs (row) from the training data.

        :param row: one individual training data with random effect covariates
        :param params: a tuple of dictionaries, which map random effects to their distribution, and sampled value nodes
        :return: BMGraph node that sums over all random effects for a given observation (i.e. a row from the training data).
        """

        if not self.random_effects:
            return self.zero
        (
            re_dist,
            re_value,
        ) = params
        re_list = []
        for re in self.random_effects:
            key = tuple(row[x] for x in re) if isinstance(re, tuple) else row[re]
            if key not in re_value[re]:
                if isinstance(re_dist[re], dict):
                    # structured prior: re_dist[re] is a dictionary, mapping r.e. levels to their sample nodes
                    # then just copy re_dist[re] to re_value[re]
                    re_value[re][key] = copy.deepcopy(re_dist[re][key])
                else:
                    re_value[re][key] = self.g.add_operator(
                        bmgraph.OperatorType.SAMPLE, [re_dist[re]]
                    )
            re_list.append(re_value[re][key])
        if len(re_list) < 2:
            return re_list[0]
        return self.g.add_operator(bmgraph.OperatorType.ADD, re_list)

    def _predict_fere_byrow(
        self, new_row: pd.Series, post_samples: pd.DataFrame
    ) -> pd.Series:
        """Generates response variable predictive distribution given new test data.

        :param new_row: one individual test data for prediction
        :param post_samples: MCMC posterior inference samples on model parameters
        :return: response variable predictive distribution
        """

        pred_val = pd.Series(0.0, index=range(post_samples.shape[0]))
        for fe in self.fixed_effects:
            try:
                x = new_row[fe]
                pred_val += x * post_samples["fixed_effect_" + str(fe)]
            except KeyError:
                logger.warning(
                    "fixed effect: %s is not available in the "
                    "posterior samples or the new data.",
                    fe,
                )
                return pd.Series(None, index=range(post_samples.shape[0]))
        for re in self.random_effects:
            try:
                re_level = (
                    tuple(new_row[x] for x in re)
                    if isinstance(re, tuple)
                    else new_row[re]
                )
                key = "re_value_" + str(re) + "_" + str(re_level)
            except KeyError:
                logger.warning(
                    "Random effect: "
                    + str(re)
                    + " is not available in the new data row:\n{r}\n".format(r=new_row)
                )
                return pd.Series(None, index=range(post_samples.shape[0]))
            try:
                pred_val += post_samples[key]
            except KeyError:
                logger.warning(
                    "Random effect level: "
                    + str(re)
                    + " = "
                    + str(re_level)
                    + " is not available in the posterior samples."
                )
                return pd.Series(None, index=range(post_samples.shape[0]))
        return pred_val
