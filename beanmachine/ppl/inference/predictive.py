# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List, Optional

from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from beanmachine.ppl.model.utils import RVIdentifier

from .single_site_ancestral_mh import SingleSiteAncestralMetropolisHastings


class Predictive(object):
    """
    Abstract class for predictive monte carlo samples.
    """

    @staticmethod
    def simulate(
        queries: List[RVIdentifier],
        posterior: Optional[MonteCarloSamples] = None,
        num_samples: int = 1,
        num_chains: int = 1,
    ) -> MonteCarloSamples:
        """
        Generates predictives from a generative model.

           queries = [likelihood(i) for i in range(10))]
           posterior = SinglesiteHamiltonianMonteCarlo(10, 0.1).infer(...)
           predictives = simulate(queries, posterior=posterior, num_samples=1000)

        To generate prior predictives,

           queries = [prior(), likelihood()]  # specify the full generative model
           predictives = simulate(queries, num_samples=1000)
           # Monte carlo samples of shape (num_pred_chains, num_pred_samples,
           # num_post_chains, num_post_samples, sample_shape)

        :param query: list of ``random_variable``s corresponding to the observations
        :param posterior: Optional ``MonteCarloSamples`` of the latent variables
        :param num_samples: number of predictive samples, defaults to 1
        :param num_chains: number of chains, defaults to 1
        :returns: monte carlo samples of the generated predictive sampled with
                  replacement
        """
        sampler = SingleSiteAncestralMetropolisHastings()
        obs = posterior.data.rv_dict if posterior else {}
        samples = sampler.infer(
            queries,
            obs,
            num_samples=num_samples,
            num_chains=num_chains,
            initialize_from_prior=True,
        )
        return samples


simulate = Predictive.simulate
