# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Union

import torch
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from beanmachine.ppl.inference.single_site_ancestral_mh import (
    SingleSiteAncestralMetropolisHastings,
)
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world import init_from_prior, RVDict, World
from torch import Tensor
from torch.distributions import Categorical
from tqdm.auto import trange


def _concat_rv_dicts(rvdict: List) -> Dict:
    out_dict = {}
    keys = list(rvdict[0].keys())
    for k in keys:
        t = []
        for x in rvdict:
            t.append(x[k])
        out_dict[k] = torch.cat(t, -1).squeeze(0)
    return out_dict


class Predictive(object):
    """
    Class for the posterior predictive distribution.
    """

    @staticmethod
    def _extract_values_from_world(
        world: World, queries: List[RVIdentifier]
    ) -> Dict[RVIdentifier, Tensor]:
        query_dict = {query: [] for query in queries}
        # Extract samples
        for query in queries:
            raw_val = world.call(query)
            if not isinstance(raw_val, torch.Tensor):
                raise TypeError(
                    "The value returned by a queried function must be a tensor."
                )
            query_dict[query].append(raw_val)
        query_dict = {node: torch.stack(val) for node, val in query_dict.items()}
        return query_dict

    @staticmethod  # noqa: C901
    def simulate(  # noqa: C901
        queries: List[RVIdentifier],
        posterior: Optional[Union[MonteCarloSamples, RVDict]] = None,
        num_samples: Optional[int] = None,
        vectorized: Optional[bool] = False,
        progress_bar: Optional[bool] = True,
    ) -> MonteCarloSamples:
        """
        Generates predictives from a generative model.

        For example::

           obs_queries = [likelihood(i) for i in range(10))]
           posterior = SinglesiteHamiltonianMonteCarlo(10, 0.1).infer(...)
           # generates one sample per world (same shape as `posterior` samples)
           predictives = simulate(obs_queries, posterior=posterior)

        To generate prior predictives::

           queries = [prior(), likelihood()]  # specify the full generative model
           # Monte carlo samples of shape (num_samples, sample_shape)
           predictives = simulate(queries, num_samples=1000)

        :param query: list of `random_variable`'s corresponding to the observations.
        :param posterior: Optional `MonteCarloSamples` or `RVDict` of the latent variables.
        :param num_samples: Number of prior predictive samples, defaults to 1. Should
            not be specified if `posterior` is specified.
        :returns: `MonteCarloSamples` of the generated predictives.
        """
        assert (
            (posterior is not None) + (num_samples is not None)
        ) == 1, "Only one of posterior or num_samples should be set."
        inference = SingleSiteAncestralMetropolisHastings()
        if posterior is not None:
            if isinstance(posterior, dict):
                posterior = MonteCarloSamples([posterior])

            obs = dict(posterior)
            if vectorized:
                sampler = inference.sampler(
                    queries, obs, num_samples, initialize_fn=init_from_prior
                )
                query_dict = Predictive._extract_values_from_world(
                    next(sampler), queries
                )

                for rvid, rv in query_dict.items():
                    if rv.dim() > 2:
                        query_dict[rvid] = rv.squeeze(0)
                post_pred = MonteCarloSamples(
                    query_dict,
                    default_namespace="posterior_predictive",
                )
                post_pred.add_groups(posterior)
                return post_pred
            else:
                # predictives are sequentially sampled
                preds = []

                for c in range(posterior.num_chains):
                    rv_dicts = []
                    for i in trange(
                        posterior.get_num_samples(),
                        desc="Samples collected",
                        disable=not progress_bar,
                    ):
                        obs = {rv: posterior.get_chain(c)[rv][i] for rv in posterior}
                        sampler = inference.sampler(
                            queries, obs, num_samples, initialize_fn=init_from_prior
                        )
                        rv_dicts.append(
                            Predictive._extract_values_from_world(
                                next(sampler), queries
                            )
                        )
                    preds.append(_concat_rv_dicts(rv_dicts))
                post_pred = MonteCarloSamples(
                    preds,
                    default_namespace="posterior_predictive",
                )
                post_pred.add_groups(posterior)
                return post_pred
        else:
            obs = {}
            predictives = []

            for _ in trange(
                num_samples, desc="Samples collected", disable=not progress_bar
            ):
                sampler = inference.sampler(
                    queries, obs, num_samples, initialize_fn=init_from_prior
                )
                query_dict = Predictive._extract_values_from_world(
                    next(sampler), queries
                )
                predictives.append(query_dict)

            rv_dict = {}
            for k in predictives:
                for rvid, rv in k.items():
                    if rvid not in rv_dict:
                        rv_dict[rvid] = []
                    if rv.dim() < 2:
                        rv = rv.unsqueeze(0)
                    rv_dict[rvid].append(rv)
            for k, v in rv_dict.items():
                rv_dict[k] = torch.cat(v, dim=1)
            prior_pred = MonteCarloSamples(
                rv_dict,
                default_namespace="prior_predictive",
            )
            return prior_pred

    @staticmethod
    def empirical(
        queries: List[RVIdentifier],
        samples: MonteCarloSamples,
        num_samples: Optional[int] = 1,
    ) -> MonteCarloSamples:
        """
        Samples from the empirical (marginal) distribution of the queried variables.

        :param queries: list of `random_variable`'s to be sampled.
        :param samples: `MonteCarloSamples` of the distribution.
        :param num_samples: Number of samples to sample (with replacement). Defaults to 1.
        :returns: `MonteCarloSamples` object containing the sampled random variables.

        """
        rv_dict = {}
        num_chains = samples.num_chains
        total_num_samples = samples.get_num_samples()
        chain_indices = Categorical(torch.ones(num_chains)).sample((num_samples,))
        sample_indices = Categorical(torch.ones(total_num_samples)).sample(
            (num_samples,)
        )
        for q in queries:
            rv_dict[q] = samples.get_variable(q, include_adapt_steps=False)[
                chain_indices, sample_indices
            ]
        return MonteCarloSamples([rv_dict])


simulate = Predictive.simulate
empirical = Predictive.empirical
