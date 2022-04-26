# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Iterator, List, Mapping, NamedTuple, Optional, Union

import arviz as az
import torch
import xarray as xr
from beanmachine.ppl.inference.utils import detach_samples, merge_dicts
from beanmachine.ppl.model.rv_identifier import RVIdentifier


RVDict = Dict[RVIdentifier, torch.Tensor]


class Samples(NamedTuple):
    samples: RVDict
    adaptive_samples: RVDict


class MonteCarloSamples(Mapping[RVIdentifier, torch.Tensor]):
    """
    Represents a view of the data representing the results of infer

    If no chain is specified, the data across all chains is accessible
    If a chain is specified, only the data from the chain will be accessible
    """

    def __init__(
        self,
        chain_results: Union[List[RVDict], RVDict],
        num_adaptive_samples: int = 0,
        logll_results: Optional[Union[List[RVDict], RVDict]] = None,
        observations: Optional[RVDict] = None,
        stack_not_cat: bool = True,
        default_namespace: str = "posterior",
    ):
        self.namespaces = {}
        self.default_namespace = default_namespace

        if default_namespace not in self.namespaces:
            self.namespaces[default_namespace] = {}

        if isinstance(chain_results, list):
            self.num_chains = len(chain_results)
            chain_results = merge_dicts(chain_results, 0, stack_not_cat)
        else:
            self.num_chains = next(iter(chain_results.values())).shape[0]
        self.num_adaptive_samples = num_adaptive_samples

        self.namespaces[default_namespace] = Samples({}, {})
        for rv, val in chain_results.items():
            self.adaptive_samples[rv] = val[:, :num_adaptive_samples]
            self.samples[rv] = val[:, num_adaptive_samples:]

        if logll_results is not None:
            if isinstance(logll_results, list):
                logll = merge_dicts(logll_results, 0, stack_not_cat)
            else:
                logll = logll_results
            self.log_likelihoods = {}
            self.adaptive_log_likelihoods = {}
            for rv, val in logll.items():
                self.adaptive_log_likelihoods[rv] = val[:, :num_adaptive_samples]
                self.log_likelihoods[rv] = val[:, num_adaptive_samples:]
        else:
            self.log_likelihoods = None
            self.adaptive_log_likelihoods = None

        self.observations = observations

        # single_chain_view is only set when self.get_chain is called
        self.single_chain_view = False

    @property
    def samples(self):
        return self.namespaces[self.default_namespace].samples

    @property
    def adaptive_samples(self):
        return self.namespaces[self.default_namespace].adaptive_samples

    def __getitem__(self, rv: RVIdentifier) -> torch.Tensor:
        """
        :param rv: random variable to view values of
        :results: samples drawn during inference for the specified variable
        """
        return self.get_variable(rv, include_adapt_steps=False)

    def __iter__(self) -> Iterator[RVIdentifier]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __str__(self) -> str:
        return str(self.samples)

    def get_chain(self, chain: int = 0) -> "MonteCarloSamples":
        """
        Return a MonteCarloSamples with restricted view to a specified chain

        :param chain: specific chainto view.
        :returns: view of the data restricted to specified chain
        """
        if self.single_chain_view:
            raise ValueError(
                "The current MonteCarloSamples object has already been"
                " restricted to a single chain"
            )
        elif chain < 0 or chain >= self.num_chains:
            raise IndexError("Please specify a valid chain")

        samples = {rv: self.get_variable(rv, True)[[chain]] for rv in self}

        if self.log_likelihoods is None:
            logll = None
        else:
            logll = {
                rv: self.get_log_likelihoods(rv, True)[[chain]]
                for rv in self.log_likelihoods
            }

        new_mcs = MonteCarloSamples(
            chain_results=samples,
            num_adaptive_samples=self.num_adaptive_samples,
            logll_results=logll,
            observations=self.observations,
            default_namespace=self.default_namespace,
        )
        new_mcs.single_chain_view = True

        return new_mcs

    def get_variable(
        self,
        rv: RVIdentifier,
        include_adapt_steps: bool = False,
        thinning: int = 1,
        namespace: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Let C be the number of chains,
        S be the number of samples

        If include_adapt_steps, S' = S.
        Else, S' = S - num_adaptive_samples.

        if no chain specified:
            samples[var] returns a Tensor of (C, S', (shape of Var))
        if a chain is specified:
            samples[var] returns a Tensor of (S', (shape of Var))

        :param rv: random variable to see samples
        :param include_adapt_steps: Indicates whether the beginning of the
            chain should be included with the healthy samples.
        :returns: samples drawn during inference for the specified variable
        """

        if not isinstance(rv, RVIdentifier):
            raise TypeError(
                "The key is required to be a random variable "
                + f"but is of type {type(rv).__name__}."
            )

        if namespace is None:
            namespace = self.default_namespace

        samples = self.namespaces[namespace].samples[rv]

        if include_adapt_steps:
            samples = torch.cat(
                [self.namespaces[namespace].adaptive_samples[rv], samples],
                dim=1,
            )

        if thinning > 1:
            samples = samples[:, ::thinning]
        if self.single_chain_view:
            samples = samples.squeeze(0)
        return samples

    def get_log_likelihoods(
        self,
        rv: RVIdentifier,
        include_adapt_steps: bool = False,
    ) -> torch.Tensor:
        """
        :returns: log_likelihoods computed during inference for the specified variable
        """

        if not isinstance(rv, RVIdentifier):
            raise TypeError(
                "The key is required to be a random variable "
                + f"but is of type {type(rv).__name__}."
            )

        logll = self.log_likelihoods[rv]

        if include_adapt_steps:
            logll = torch.cat([self.adaptive_log_likelihoods[rv], logll], dim=1)

        if self.single_chain_view:
            logll = logll.squeeze(0)
        return logll

    def get(
        self,
        rv: RVIdentifier,
        default: Any = None,
        chain: Optional[int] = None,
        include_adapt_steps: bool = False,
        thinning: int = 1,
    ):
        """
        Return the value of the random variable if rv is in the dictionary, otherwise
        return the default value. This method is analogous to Python's dict.get(). The
        chain and include_adapt_steps parameters serve the same purpose as in get_chain
        and get_variable.
        """
        if rv not in self.samples:
            return default

        if chain is None:
            samples = self
        else:
            samples = self.get_chain(chain)

        return samples.get_variable(rv, include_adapt_steps, thinning)

    def get_num_samples(self, include_adapt_steps: bool = False) -> int:
        """
        :returns: the number of samples run during inference
        """
        num_samples = next(iter(self.samples.values())).shape[1]
        if include_adapt_steps:
            return num_samples + self.num_adaptive_samples
        return num_samples

    def to_xarray(self, include_adapt_steps: bool = False) -> xr.Dataset:
        """
        Return an xarray.Dataset from MonteCarloSamples.
        """
        inference_data = self.to_inference_data(include_adapt_steps)
        if not include_adapt_steps:
            return inference_data["posterior"]
        else:
            return xr.concat(
                [inference_data["warmup_posterior"], inference_data["posterior"]],
                dim="draw",
            )

    def add_groups(self, mcs: "MonteCarloSamples"):
        if self.observations is None:
            self.observations = mcs.observations

        if self.log_likelihoods is None:
            self.log_likelihoods = mcs.log_likelihoods

        if self.adaptive_log_likelihoods is None:
            self.adaptive_log_likelihoods = mcs.adaptive_log_likelihoods

        for n in mcs.namespaces:
            if n not in self.namespaces:
                self.namespaces[n] = mcs.namespaces[n]

    def to_inference_data(self, include_adapt_steps: bool = False) -> az.InferenceData:
        """
        Return an az.InferenceData from MonteCarloSamples.
        """

        if "posterior" in self.namespaces:
            posterior = detach_samples(self.namespaces["posterior"].samples)
            if self.num_adaptive_samples > 0:
                warmup_posterior = detach_samples(
                    self.namespaces["posterior"].adaptive_samples
                )
            else:
                warmup_posterior = None
        else:
            posterior = None
            warmup_posterior = None

        if self.num_adaptive_samples > 0:
            warmup_log_likelihood = self.adaptive_log_likelihoods
            if warmup_log_likelihood is not None:
                warmup_log_likelihood = detach_samples(warmup_log_likelihood)
        else:
            warmup_log_likelihood = None

        if "posterior_predictive" in self.namespaces:
            posterior_predictive = detach_samples(
                self.namespaces["posterior_predictive"].samples
            )
            if self.num_adaptive_samples > 0:
                warmup_posterior_predictive = detach_samples(
                    self.namespaces["posterior"].adaptive_samples
                )
            else:
                warmup_posterior_predictive = None
        else:
            posterior_predictive = None
            warmup_posterior_predictive = None

        if "prior_predictive" in self.namespaces:
            prior_predictive = detach_samples(
                self.namespaces["prior_predictive"].samples
            )
        else:
            prior_predictive = None

        if self.log_likelihoods is not None:
            log_likelihoods = detach_samples(self.log_likelihoods)
        else:
            log_likelihoods = None
        if self.observations is not None:
            observed_data = detach_samples(self.observations)
        else:
            observed_data = None

        return az.from_dict(
            posterior=posterior,
            warmup_posterior=warmup_posterior,
            posterior_predictive=posterior_predictive,
            warmup_posterior_predictive=warmup_posterior_predictive,
            prior_predictive=prior_predictive,
            save_warmup=include_adapt_steps,
            warmup_log_likelihood=warmup_log_likelihood,
            log_likelihood=log_likelihoods,
            observed_data=observed_data,
        )
