# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import re as regx
import time
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import beanmachine.graph as bmgraph
import beanmachine.ppl.diagnostics.common_statistics as bm_diag_util
import numpy as np
import pandas as pd

from .configs import InferConfig, ModelConfig


import torch  # usort: skip # noqa: F401


logger = logging.getLogger("hme")


class AbstractModel(object, metaclass=ABCMeta):
    """
    An abstract class for Bayesian graphical models using BMGraph.
    """

    def __init__(self, data: pd.DataFrame, model_config: ModelConfig) -> None:
        self.data = data
        self.model_config = model_config
        self.queries = {}
        self.query_map = {}
        self.g = bmgraph.Graph()

    @staticmethod
    def parse_formula(formula: str) -> Tuple[List, List]:
        def make_joint_effect(effect):
            eff_split = effect.split("+")
            if len(eff_split) == 1:
                return effect
            else:
                return tuple(eff_split)

        def expand_nested_effects(effect):
            if len(effect) == 1:
                return [make_joint_effect(effect[0])]
            expanded_effects = [[effect[0]]]
            i = 1
            while i < len(effect):
                new_effect = expanded_effects[-1].copy()
                new_effect.append(effect[i])
                expanded_effects.append(new_effect)
                i += 1
            expanded_effects = [tuple(a) for a in expanded_effects]
            expanded_effects[0] = effect[0]
            return expanded_effects

        lhs, rhs = formula.split("~")
        # outcome = lhs.strip()
        all_re = [
            rand_eff.strip().split("|")[1].strip().split("/")
            for rand_eff in regx.findall(r"\((.*?)\)", rhs)
        ]
        all_fe = [
            fixed_eff.strip()
            for fixed_eff in regx.split(r"[\s]*\+[\s]*\(", rhs)
            if "(" not in fixed_eff and ")" not in fixed_eff
        ]
        if all_fe:
            fixed_effects = [fe.strip() for fe in all_fe[0].split("+")]
        else:
            fixed_effects = []
        random_effects = []
        for effect in all_re:
            random_effects += expand_nested_effects(effect)
        return fixed_effects, random_effects

    @abstractmethod
    def build_graph(self) -> None:
        """
        This method creates a bmgraph.Graph member for the model.
        """
        pass

    def set_queries(self, manual_queries: Optional[Dict[str, Any]] = None) -> None:
        queries = manual_queries if manual_queries else self.queries
        query_map = {}
        self._add_query(None, queries, query_map)
        self.query_map = query_map

    def _add_query(
        self, prev_key: Any, queries: Any, query_map: Dict[str, int]
    ) -> None:
        if isinstance(queries, dict):
            try:
                kv_pairs = sorted(queries.items())
            except TypeError:
                kv_pairs = queries.items()
            for key, value in kv_pairs:
                new_key = str(prev_key) + "_" + str(key) if prev_key else str(key)
                self._add_query(new_key, value, query_map)
        else:
            query_map[str(prev_key)] = self.g.query(queries)

    def infer(self, infer_config: InferConfig) -> Tuple[pd.DataFrame]:
        t0 = time.time()
        if not self.query_map:
            self.set_queries()
        iconf = bmgraph.InferConfig()
        iconf.keep_log_prob = infer_config.keep_logprob
        bmg_result = self.g.infer(
            infer_config.n_iter + infer_config.n_warmup,
            bmgraph.InferenceType.NMC,
            infer_config.seed,
            infer_config.n_chains,
            iconf,
        )
        keep_after = 0 if infer_config.keep_warmup else infer_config.n_warmup
        posterior_samples = self._to_dataframe(bmg_result, keep_after)
        posterior_diagnostics = self._get_bmg_diagnostics(bmg_result, keep_after)
        t_used = time.time() - t0
        logger.info(f"The model fitted in {round(t_used/60., 1)} minutes.")
        return posterior_samples, posterior_diagnostics

    def _to_dataframe(self, dat_list: List, keep_after: int) -> pd.DataFrame:
        df_list = []
        i = 0
        for chain_i in dat_list:
            df = (
                pd.DataFrame(chain_i, columns=list(self.query_map.keys())) * 1.0
            )  # multiply by 1.0 to convert bool-valued h to float
            df["iter"] = pd.Series(range(len(chain_i))) + 1
            df["chain"] = i
            i += 1
            df_list.append(df[keep_after:])
        result_df = pd.concat(df_list, axis=0).reset_index(drop=True)
        return result_df

    def _get_bmg_diagnostics(self, samples: List, keep_after: int) -> pd.DataFrame:

        # conversion to torch
        samples = torch.tensor(samples)[:, keep_after:, :]

        n_chain = samples.shape[0]
        bmg_neff = bm_diag_util.effective_sample_size(samples)
        bmg_rhat = bm_diag_util.split_r_hat(samples)
        bmg_mean = bm_diag_util.mean(samples)

        posterior_diagnostics = pd.DataFrame()
        posterior_diagnostics["names"] = np.array(list(self.query_map.keys()))
        posterior_diagnostics["Mean"] = np.array(
            [bmg_mean[self.query_map[k]] for k in posterior_diagnostics["names"].values]
        )
        posterior_diagnostics["Acceptance"] = np.array(
            [
                len(samples[:, :, self.query_map[k]].reshape(-1).unique())
                / len(samples[:, :, self.query_map[k]].reshape(-1))
                for k in posterior_diagnostics["names"].values
            ]
        )
        posterior_diagnostics["N_Eff"] = np.array(
            [bmg_neff[self.query_map[k]] for k in posterior_diagnostics["names"].values]
        )
        if n_chain > 1:
            posterior_diagnostics["R_hat"] = np.array(
                [
                    bmg_rhat[self.query_map[k]]
                    for k in posterior_diagnostics["names"].values
                ]
            )
        else:
            logger.warning("Convergence diagnostic, R_hat, requires more than 1 chain.")

        logger.warning(
            "Generated acceptance rates are only valid for continuous random variables."
        )
        return posterior_diagnostics

    def _customize_priors(self) -> None:
        """
        Create customized prior dist based on model_config.priors. e.g.
        {"fe": PriorConfig('normal', 'real', [0.0, 1.0])}
        """
        self.customized_priors = {}
        # FIXME: update the parser of priors
        # for key, val in self.model_config.priors.items():
        #     param_list = []
        #     for param in val.parameters:
        #         pass
        #     self.customized_priors[key] = self.g.add_distribution(
        #         val.distribution, val.support, param_list
        #     )
