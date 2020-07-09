# Copyright (c) Facebook, Inc. and its affiliates.
from collections import defaultdict
from typing import List, Optional

import torch
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from beanmachine.ppl.model.utils import RVIdentifier

from .single_site_ancestral_mh import SingleSiteAncestralMetropolisHastings


class Predictive(object):
    """
    Class for the posterior predictive distribution.
    """

    @staticmethod
    def simulate(
        queries: List[RVIdentifier],
        posterior: Optional[MonteCarloSamples] = None,
        num_samples: Optional[int] = None,
    ):
        """
        Generates predictives from a generative model.

           obs_queries = [likelihood(i) for i in range(10))]
           posterior = SinglesiteHamiltonianMonteCarlo(10, 0.1).infer(...)
           # generates one sample per world (same shape as ``posterior`` samples)
           predictives = simulate(obs_queries, posterior=posterior)

        To generate prior predictives,

           queries = [prior(), likelihood()]  # specify the full generative model
           # Monte carlo samples of shape (num_samples, sample_shape)
           predictives = simulate(queries, num_samples=1000)

        :param query: list of ``random_variable``s corresponding to the observations.
        :param posterior: Optional ``MonteCarloSamples`` of the latent variables.
        :param num_samples: Number of prior predictive samples, defaults to 1. Should
                            not be specified if ``posterior`` is specified.
        :returns: ``MonteCarloSamples`` of the generated predictives.
        """
        assert (
            (posterior is not None) + (num_samples is not None)
        ) == 1, "Only one of posterior or num_samples should be set."
        sampler = SingleSiteAncestralMetropolisHastings()
        if posterior:
            obs = posterior.data.rv_dict
            # predictives are jointly sampled
            sampler.queries_ = queries
            sampler.observations_ = obs
            query_dict = sampler._infer(1, initialize_from_prior=True)
            sampler.reset()
            for rvid, rv in query_dict.items():
                if rv.dim() > 2:
                    query_dict[rvid] = rv.squeeze(0)
            return MonteCarloSamples(query_dict)
        else:
            obs = {}
            predictives = []

            # pyre-fixme
            for _ in range(num_samples):
                sampler.queries_ = queries
                sampler.observations_ = obs
                query_dict = sampler._infer(1, initialize_from_prior=True)
                predictives.append(query_dict)
                sampler.reset()

            rv_dict = defaultdict(list)
            for k in predictives:
                for rvid, rv in k.items():
                    if rv.dim() < 2:
                        rv = rv.unsqueeze(0)
                    rv_dict[rvid].append(rv)
            for k, v in rv_dict.items():
                # pyre-fixme
                rv_dict[k] = torch.cat(v, dim=1)
            # pyre-fixme
            return MonteCarloSamples(dict(rv_dict))


simulate = Predictive.simulate
