# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List, Optional

import torch
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from beanmachine.ppl.legacy.inference.single_site_ancestral_mh import (
    SingleSiteAncestralMetropolisHastings,
)
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from torch.distributions import Categorical


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

    @staticmethod  # noqa: C901
    def simulate(  # noqa: C901
        queries: List[RVIdentifier],
        posterior: Optional[MonteCarloSamples] = None,
        num_samples: Optional[int] = None,
        vectorized: Optional[bool] = False,
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
        :param posterior: Optional `MonteCarloSamples` of the latent variables.
        :param num_samples: Number of prior predictive samples, defaults to 1. Should
            not be specified if `posterior` is specified.
        :returns: `MonteCarloSamples` of the generated predictives.
        """
        assert (
            (posterior is not None) + (num_samples is not None)
        ) == 1, "Only one of posterior or num_samples should be set."
        sampler = SingleSiteAncestralMetropolisHastings()
        if posterior:
            n_warmup = posterior.num_adaptive_samples
            # drop the warm up samples
            obs = {k: v[:, n_warmup:] for k, v in posterior.items()}
            if vectorized:
                # predictives are jointly sampled
                sampler.queries_ = queries
                sampler.observations_ = obs
                try:
                    query_dict = sampler._infer(1, initialize_from_prior=True)
                finally:
                    sampler.reset()
                for rvid, rv in query_dict.items():
                    if rv.dim() > 2:
                        query_dict[rvid] = rv.squeeze(0)
                return MonteCarloSamples(query_dict)
            else:
                # predictives are sequentially sampled
                preds = []

                for c in range(posterior.num_chains):
                    rv_dicts = []
                    for i in range(posterior.get_num_samples()):
                        obs = {rv: posterior.get_chain(c)[rv][i] for rv in posterior}
                        sampler.queries_ = queries
                        sampler.observations_ = obs
                        try:
                            rv_dicts.append(
                                sampler._infer(1, initialize_from_prior=True)
                            )
                        finally:
                            sampler.reset()
                    preds.append(_concat_rv_dicts(rv_dicts))
                return MonteCarloSamples(preds)
        else:
            obs = {}
            predictives = []

            # pyre-fixme
            for _ in range(num_samples):
                sampler.queries_ = queries
                sampler.observations_ = obs
                try:
                    query_dict = sampler._infer(1, initialize_from_prior=True)
                finally:
                    sampler.reset()
                predictives.append(query_dict)

            rv_dict = defaultdict(list)
            for k in predictives:
                for rvid, rv in k.items():
                    if rv.dim() < 2:
                        rv = rv.unsqueeze(0)
                    rv_dict[rvid].append(rv)
            for k, v in rv_dict.items():
                # pyre-ignore
                rv_dict[k] = torch.cat(v, dim=1)
            # pyre-fixme
            return MonteCarloSamples(dict(rv_dict))

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
