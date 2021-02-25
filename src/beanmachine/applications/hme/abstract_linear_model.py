# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple

import beanmachine.graph as bmgraph
import pandas as pd

from .abstract_model import AbstractModel
from .configs import ModelConfig


logger = logging.getLogger("hme")


class AbstractLinearModel(AbstractModel, metaclass=ABCMeta):
    """
    An abstract class for creating linear mixed effects model using BMGraph.
    """

    def __init__(self, data: pd.DataFrame, model_config: ModelConfig) -> None:
        super().__init__(data, model_config)
        self.fixed_effects = []
        self.random_effects = []

    @abstractmethod
    def _add_observation_byrow(self, index: int, row: pd.Series, fere_i: int) -> None:
        """
        This method defines the conditional distribution of the observation node
        given the linear component node: fere_i
        """
        pass

    def _set_priors(self) -> None:
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

    def _initialize_fixed_effect_nodes(self) -> Dict[str, int]:
        fe_prior = (
            self.customized_priors["fe"]
            if "fe" in self.customized_priors
            else self.normal_prior
        )
        fixed_effects_params = {
            fe: self.g.add_operator(bmgraph.OperatorType.SAMPLE, [fe_prior])
            for fe in self.fixed_effects
        }
        return fixed_effects_params

    def _initialize_random_effect_nodes(self, use_t_random_effect: bool) -> Tuple:
        re_scale_prior = (
            self.customized_priors["re_scale"]
            if "re_scale" in self.customized_priors
            else self.halfcauchy_prior
        )
        re_scale = {
            re: self.g.add_operator(bmgraph.OperatorType.SAMPLE, [re_scale_prior])
            for re in self.random_effects
        }
        if use_t_random_effect:
            # FIXME: add truncated normal to support dof prior
            re_dof_prior = (
                self.customized_priors["re_dof"]
                if "re_dof" in self.customized_priors
                else self.halfcauchy_prior
            )
            re_dof = {
                re: self.g.add_operator(bmgraph.OperatorType.SAMPLE, [re_dof_prior])
                for re in self.random_effects
            }
            re_dist = {
                re: self.g.add_distribution(
                    bmgraph.DistributionType.STUDENT_T,
                    bmgraph.AtomicType.REAL,
                    [re_dof[re], self.zero, re_scale[re]],
                )
                for re in self.random_effects
            }
        else:
            re_dof = None
            re_dist = {
                re: self.g.add_distribution(
                    bmgraph.DistributionType.NORMAL,
                    bmgraph.AtomicType.REAL,
                    [self.zero, re_scale[re]],
                )
                for re in self.random_effects
            }
        re_value = {re: {} for re in self.random_effects}
        return re_dof, re_scale, re_dist, re_value

    def _add_fixed_effects_byrow(self, row: pd.Series, params: Dict[str, int]) -> int:
        if not self.fixed_effects:
            return self.zero
        fe_list = []
        for fe in self.fixed_effects:
            if fe == "1":
                fe_list.append(params[fe])
            else:
                # FIXME: add support for categorical data
                x = self.g.add_constant(row[fe])
                x_param = self.g.add_operator(
                    bmgraph.OperatorType.MULTIPLY, [x, params[fe]]
                )
                fe_list.append(x_param)
        if len(fe_list) < 2:
            return fe_list[0]
        return self.g.add_operator(bmgraph.OperatorType.ADD, fe_list)

    def _add_random_effects_byrow(self, row: pd.Series, params: Tuple) -> int:
        if not self.random_effects:
            return self.zero
        re_dist, re_value = params
        re_list = []
        for re in self.random_effects:
            key = tuple(row[x] for x in re) if isinstance(re, tuple) else row[re]
            if key not in re_value[re]:
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
        pred_val = pd.Series(0.0, index=range(post_samples.shape[0]))
        for fe in self.fixed_effects:
            try:
                if fe == "1":
                    pred_val += post_samples["fixed_effect_1"]
                else:
                    x = new_row[fe]
                    pred_val += x * post_samples["fixed_effect_" + str(fe)]
            except KeyError:
                logger.warning(
                    "fixed_effect: "
                    + str(fe)
                    + " is not available in "
                    + "the posterior samples or the new data."
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
