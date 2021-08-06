# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import re as regx
import time
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import beanmachine.graph as bmgraph
import beanmachine.ppl.diagnostics.common_statistics as bm_diag_util
import numpy as np
import pandas as pd
from patsy import ModelDesc, dmatrix, dmatrices, build_design_matrices
import torch  # usort: skip # noqa: F401

from .configs import InferConfig, ModelConfig, PriorConfig
from .patsy_mixed import evaluate_formula, RandomEffectsTerm
from .priors import DIST_TYPE_DICT, SAMPLE_TYPE_DICT, PARAM_TYPE_DICT, ATOMIC_TYPE_DICT


logger = logging.getLogger("hme")


class AbstractModel(object, metaclass=ABCMeta):
    """An abstract class for Bayesian graphical models using BMGraph.

    :param data: observed train data
    :type data: class:`pd.DataFrame`
    :param model_config: model configuration parameters
    :type model_config: class:`ModelConfig`
    """

    def __init__(self, data: pd.DataFrame, model_config: ModelConfig) -> None:
        self.data = data
        self.model_config = model_config
        self.queries = {}
        self.query_map = {}
        self.design_infos = []
        self.g = bmgraph.Graph()

    @staticmethod
    def parse_formula(formula: str) -> Tuple[List, List]:
        """Extracts fixed and/or random effects from given statistical formula.

        :param formula: statistical formula establishing the relationship between response and predictor variables
        :type formula: str
        :return: A tuple of fixed and/or random effects lists
        :rtype: (list, list)
        """

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
        """Creates a bmgraph.Graph() member for the model."""

        pass

    def parse_formula_patsy(self, formula: str) -> Tuple[List, List]:
        """Parses statistical formula as well as performs data pre-processing given statistical formula.

        :param formula: statistical formula establishing the relationship between response and predictor variables
        :type formula: str
        :return: A tuple of fixed and/or random effects lists
        :rtype: (list, list)
        """

        model_desc = evaluate_formula(formula)

        # outcome consistency check:
        # if outcome is specified by the formula,
        # then we need to compare it with self.model_config.mean_regression.outcome
        if model_desc.lhs_termlist:
            outcome = model_desc.lhs_termlist[0].factors[0].code
            if outcome != self.model_config.mean_regression.outcome:
                raise ValueError(
                    f"Inconsistent outcome variable encountered! Formula: {outcome}; RegressionConfig: {self.model_config.mean_regression.outcome}."
                )

        fe_termlists, re_termlists = [], []
        for term in model_desc.rhs_termlist:
            if isinstance(term, RandomEffectsTerm):
                re_termlists.append(term)
            else:
                fe_termlists.append(term)

        fe_desc = ModelDesc(
            lhs_termlist=model_desc.lhs_termlist, rhs_termlist=fe_termlists
        )

        # TODO: add support for data pre-processing on random effects
        if model_desc.lhs_termlist:
            y_dm, X_dm = dmatrices(fe_desc, self.data, return_type="dataframe")
            self.design_infos += [
                y_dm.design_info,
                X_dm.design_info,
            ]  # for data pre-processing as well as stateful transforms on test data
        else:
            X_dm = dmatrix(fe_desc, self.data, return_type="dataframe")
            self.design_infos += [X_dm.design_info]

        fixed_effects = X_dm.design_info.column_names

        # TODO: add support for random slope
        # for now, asssume ret.expr.rhs_termlist = [Term([])], i.e., random intercept (1|...)
        random_effects = []
        for ret in re_termlists:
            re_termlist = ret.factor.rhs_termlist
            re_factorlist = [term.factors for term in re_termlist]

            for re_factor_tuple in re_factorlist:
                re_key = tuple(re_factor.code for re_factor in re_factor_tuple)
                if len(re_key) == 1:
                    random_effects.append(re_key[0])
                else:
                    random_effects.append(re_key)

        return fixed_effects, random_effects

    def _preprocess_data(self) -> None:
        """Performs data pre-processing: including one-hot encoding on categorical fixed effects,
        as well as user-specified transformations, e.g., centering and standardization, on the outcome
        and/or predictor variables, also concatenates pre-processed data to self.data.
        """
        data_dm = pd.concat(
            build_design_matrices(
                self.design_infos, self.data, return_type="dataframe"
            ),
            axis=1,
        )
        self.preprocessed_data = pd.concat([self.data, data_dm], axis=1).loc[
            :, lambda df: np.logical_not(df.columns.duplicated())
        ]

    def set_queries(self, manual_queries: Optional[Dict[str, Any]] = None) -> None:
        """Sets query for the model. Only posterior samples for the queried random variables (parameters) will be returned.

        :param manual_queries: user-specified query, i.e., a list of model parameters user is interested to evaluate
        :type manual_queries: dict, optional
        """
        queries = manual_queries if manual_queries else self.queries
        query_map = {}
        self._add_query(None, queries, query_map)
        self.query_map = query_map

    def _add_query(
        self, prev_key: Any, queries: Any, query_map: Dict[str, int]
    ) -> None:
        """Helper function of set_queries.

        :param prev_key: None, or model parameter names of fixed or random effects
        :type prev_key: any
        :param queries: a mapping from model parameter to its BMGraph node, or just node itself
        :type queries: any
        :param query_map: a mapping from model parameter to query-index specified in the graph
        :type query_map: dict
        """
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
        """Performs MCMC posterior inference on model parameters.

        :param infer_config: configuration settings of posterior inference
        :type infer_config: class:`InferConfig`
        :return: posterior samples and their diagnostic summary statistics
        :rtype: (class:`pd.DataFrame`, class:`pd.DataFrame`)
        """

        t0 = time.time()
        if not self.query_map:
            self.set_queries()
        iconf = bmgraph.InferConfig()
        iconf.keep_log_prob = infer_config.keep_logprob
        iconf.num_warmup = infer_config.n_warmup
        iconf.keep_warmup = infer_config.keep_warmup
        bmg_result = self.g.infer(
            infer_config.n_iter,
            bmgraph.InferenceType.NMC,
            infer_config.seed,
            infer_config.n_chains,
            iconf,
        )
        posterior_samples = self._to_dataframe(bmg_result)
        posterior_diagnostics = self._get_bmg_diagnostics(bmg_result)
        t_used = time.time() - t0
        logger.info(f"The model fitted in {round(t_used/60., 1)} minutes.")
        return posterior_samples, posterior_diagnostics

    def _to_dataframe(self, dat_list: List) -> pd.DataFrame:
        """Stacks posterior sample lists across different chains and transforms them into a dataframe.

        :param dat_list: posterior samples in multiple chains generated from `bmgraph.infer()`
        :type dat_list: list
        :return: stacked posterior samples with corresponding parameter names
        :rtype: class:`pd.DataFrame`
        """

        df_list = []
        i = 0
        for chain_i in dat_list:
            df = (
                pd.DataFrame(chain_i, columns=list(self.query_map.keys())) * 1.0
            )  # multiply by 1.0 to convert bool-valued h to float
            df["iter"] = pd.Series(range(len(chain_i))) + 1
            df["chain"] = i
            i += 1
            df_list.append(df)
        result_df = pd.concat(df_list, axis=0).reset_index(drop=True)
        return result_df

    def _get_bmg_diagnostics(self, samples: List) -> pd.DataFrame:
        """Checks convergence of MCMC posterior inference results and returns diagnostic summary statistics.

        :param samples: posterior samples list generated from `bmgraph.infer()`
        :type samples: list
        :return: diagnostic summary statistics, including effective sample size, R_hat (when n_chain > 1), sample mean, and acceptance rates (for continuous R.V.)
        :rtype: class:`pd.DataFrame`
        """

        # conversion to torch
        samples = torch.tensor(samples)

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
        """Create customized prior dist based on model_config.priors. e.g.
        {"fe": PriorConfig('normal', [0.0, 1.0])}
        {"re": PriorConfig('t', [PriorConfig(), 0.0, PriorConfig()])}
        """
        self.customized_priors = {}

        for param, prior_config in self.model_config.priors.items():
            if param == "fixed_effects":
                # universal prior for all fixed effects
                if isinstance(prior_config, PriorConfig):
                    self.default_priors[param] = self._generate_prior_node(prior_config)

                # customized priors for certain fixed effects
                else:
                    self.customized_priors[param] = {}
                    for key, val in prior_config.items():
                        self.customized_priors[param][key] = self._generate_prior_node(
                            val
                        )

            elif param == "random_effects":
                # universal prior for all random effects
                if isinstance(prior_config, PriorConfig):
                    self.default_priors[param] = self._interpret_re_prior_config(
                        prior_config, param
                    )

                # customized priors for certain random effects
                else:
                    self.customized_priors[param] = defaultdict(tuple)
                    for key, val in prior_config.items():
                        self.customized_priors[param][
                            key
                        ] = self._interpret_re_prior_config(val, key)

            # -------- prob_h, prob_sign ----------
            else:
                self.customized_priors[param] = self._generate_prior_node(prior_config)

    def _generate_prior_node(self, prior_config: PriorConfig) -> int:
        """Generates a BMGraph distribution node based on the distribution description.

        :param prior_config: a distribution configuration including type and params
        :return: BMGraph distribution node for some fixed effect
        """

        dist_type = DIST_TYPE_DICT[prior_config.distribution]
        sample_type = ATOMIC_TYPE_DICT[SAMPLE_TYPE_DICT[prior_config.distribution]]

        if len(PARAM_TYPE_DICT[prior_config.distribution]) != len(
            prior_config.parameters
        ):
            raise ValueError(
                f"Number of {prior_config.distribution} distribution parameters doesn't match!"
                f"Given: {len(prior_config.parameters)}; Expected: {len(PARAM_TYPE_DICT[prior_config.distribution])}."
            )

        param_list = []
        for idx, param in enumerate(prior_config.parameters):
            param_type = list(PARAM_TYPE_DICT[prior_config.distribution].values())[idx]
            # constant param
            param_list.append(self._generate_const_node(param, param_type))

        # relevant error messages will be raised by bmgraph if any
        return self.g.add_distribution(
            dist_type,
            sample_type,
            param_list,
        )

    def _generate_const_node(
        self, param_value: float or List[List[float]], param_type: str
    ) -> int:
        """Generates a BMGraph constant node based on parameter value and type.

        :param param_value: parameter value, can be either a float number of a one-column simplex
        :param param_type: parameter type
        :return: BMGraph constant node of the given value and type
        """

        if param_type == "real" or param_type == "natural":
            return self.g.add_constant(param_value)
        elif param_type == "pos_real":
            return self.g.add_constant_pos_real(param_value)
        elif param_type == "prob":
            return self.g.add_constant_probability(param_value)
        elif param_type == "simplex":  # param_value is a 1xn matrix in this case
            return self.g.add_constant_col_simplex_matrix(param_value)
        else:
            logger.warning(
                "Parameter type '{s}' is not supported!".format(s=param_type)
            )

    def _interpret_re_prior_config(
        self,
        re_prior_config: PriorConfig,
        re_key: str,
    ) -> Tuple[int, dict]:  # (dist: int, param: dict)
        """Generates a BMGraph distribution node for the given random effect as well as hyper-prior samples of parameters within.

        :param re_prior_config: a distribution configuration including type and parameters, where parameters can either be real
            values or another `PriorConfig` instance
        :param re_key: variable name of the random effect, being "fixed_effects" implies universal priors for all random effects
        :return: BMGraph distribution node for the random effect as well as a dictionary, which maps hyper-parameters of the distribution
            to their samples
        """

        hyper_param_queries = {}
        param_list = []
        for idx, param in enumerate(re_prior_config.parameters):
            param_name = list(PARAM_TYPE_DICT[re_prior_config.distribution])[idx]
            param_type = list(PARAM_TYPE_DICT[re_prior_config.distribution].values())[
                idx
            ]

            # hyper-prior on param
            if isinstance(param, PriorConfig):
                # sanity check: hyper-prior return type
                if SAMPLE_TYPE_DICT[param.distribution] != param_type:
                    raise TypeError(
                        "Random effect: '{r}' | Prior distribution: '{d}' | Parameter: '{p}' has inconsistent value type!\n"
                        "Expected: '{e}'; Hyper-prior returns: '{h}'.".format(
                            d=re_prior_config.distribution,
                            r=re_key,
                            p=param_name,
                            e=param_type,
                            h=SAMPLE_TYPE_DICT[param.distribution],
                        )
                    )

                param_dist = self._generate_prior_node(param)
                param_sample = self.g.add_operator(
                    bmgraph.OperatorType.SAMPLE, [param_dist]
                )

                hyper_param_queries[param_name] = param_sample
                param_list.append(param_sample)
            # constant param
            else:
                param_list.append(self._generate_const_node(param, param_type))

        return (
            self.g.add_distribution(
                DIST_TYPE_DICT[re_prior_config.distribution],
                ATOMIC_TYPE_DICT[SAMPLE_TYPE_DICT[re_prior_config.distribution]],
                param_list,
            ),
            hyper_param_queries,
        )
