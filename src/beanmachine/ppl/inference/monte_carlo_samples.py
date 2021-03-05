# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Any, Dict, Iterator, List, Mapping, Optional, Union

import arviz as az
import torch
import xarray as xr
from beanmachine.ppl.model.rv_identifier import RVIdentifier


RVDict = Dict[RVIdentifier, torch.Tensor]


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
    ):
        if isinstance(chain_results, list):
            self.num_chains = len(chain_results)
            chain_results = merge_dicts(chain_results)
        else:
            self.num_chains = next(iter(chain_results.values())).shape[0]
        self.num_adaptive_samples = num_adaptive_samples

        self.adaptive_samples = {}
        self.samples = {}
        for rv, val in chain_results.items():
            self.adaptive_samples[rv] = val[:, :num_adaptive_samples]
            self.samples[rv] = val[:, num_adaptive_samples:]

        # single_chain_view is only set when self.get_chain is called
        self.single_chain_view = False

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
        new_mcs = MonteCarloSamples(samples, self.num_adaptive_samples)
        new_mcs.single_chain_view = True

        return new_mcs

    def get_variable(
        self, rv: RVIdentifier, include_adapt_steps: bool = False
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

        samples = self.samples[rv]

        if include_adapt_steps:
            samples = torch.cat([samples, self.adaptive_samples[rv]], dim=1)

        if self.single_chain_view:
            return samples.squeeze(0)
        else:
            return samples

    def get(
        self,
        rv: RVIdentifier,
        default: Any = None,
        chain: Optional[int] = None,
        include_adapt_steps: bool = False,
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

        return samples.get_variable(rv, include_adapt_steps)

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
        inferece_data = self.to_inference_data(include_adapt_steps)
        if not include_adapt_steps:
            # pyre-ignore
            return inferece_data.posterior
        else:
            return xr.concat(
                # pyre-ignore
                [inferece_data.warmup_posterior, inferece_data.posterior],
                dim="draw",
            )

    def to_inference_data(self, include_adapt_steps: bool = False) -> az.InferenceData:
        """
        Return an az.InferenceData from MonteCarloSamples.
        """
        return az.from_dict(
            posterior=self.samples,
            warmup_posterior=self.adaptive_samples,
            save_warmup=include_adapt_steps,
        )


def merge_dicts(dicts: List[RVDict], dim: int = 0) -> RVDict:
    """
    A helper function that merge multiple dicts of samples into a single dictionary,
    stacking across a new dimension
    """
    rv_keys = set().union(*(rv_dict.keys() for rv_dict in dicts))
    for idx, d in enumerate(dicts):
        if not rv_keys.issubset(d.keys()):
            raise ValueError(f"{rv_keys - d.keys()} are missing in dict {idx}")

    return {rv: torch.stack([d[rv] for d in dicts], dim=dim) for rv in rv_keys}
