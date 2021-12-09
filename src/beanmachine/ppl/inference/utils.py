# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from enum import Enum
from typing import Dict, List

import numpy.random
import torch
from beanmachine.ppl.model.rv_identifier import RVIdentifier


RVDict = Dict[RVIdentifier, torch.Tensor]

# Detect and report if a user fails to meet the inference contract.
def _verify_queries(queries: List[RVIdentifier]) -> None:
    if not isinstance(queries, list):
        t = type(queries).__name__
        raise TypeError(
            f"Parameter 'queries' is required to be a list but is of type {t}."
        )

    for query in queries:
        if not isinstance(query, RVIdentifier):
            t = type(query).__name__
            raise TypeError(
                f"A query is required to be a random variable but is of type {t}."
            )
        for arg in query.arguments:
            if isinstance(arg, RVIdentifier):
                raise TypeError(
                    "The arguments to a query must not be random variables."
                )


def _verify_observations(
    observations: Dict[RVIdentifier, torch.Tensor], must_be_rv: bool
) -> None:
    if not isinstance(observations, dict):
        t = type(observations).__name__
        raise TypeError(
            f"Parameter 'observations' is required to be a dictionary but is of type {t}."
        )

    for rv, value in observations.items():
        if not isinstance(rv, RVIdentifier):
            t = type(rv).__name__
            raise TypeError(
                f"An observation is required to be a random variable but is of type {t}."
            )
        if not isinstance(value, torch.Tensor):
            t = type(value).__name__
            raise TypeError(
                f"An observed value is required to be a tensor but is of type {t}."
            )
        if must_be_rv and rv.is_functional:
            raise TypeError(
                "An observation must observe a random_variable, not a functional."
            )
        for arg in rv.arguments:
            if isinstance(arg, RVIdentifier):
                raise TypeError(
                    "The arguments to an observation must not be random variables."
                )


def _verify_queries_and_observations(
    queries: List[RVIdentifier],
    observations: Dict[RVIdentifier, torch.Tensor],
    observations_must_be_rv: bool,
) -> None:
    _verify_queries(queries)
    _verify_observations(observations, observations_must_be_rv)


class VerboseLevel(Enum):
    """
    Enum class which is used to set how much output is printed during inference.
    LOAD_BAR enables tqdm for full inference loop.
    """

    OFF = 0
    LOAD_BAR = 1


def safe_log_prob_sum(distrib, value: torch.Tensor) -> torch.Tensor:
    "Computes log_prob, converting out of support exceptions to -Infinity."
    try:
        return distrib.log_prob(value).sum()
    except (RuntimeError, ValueError) as e:
        if not distrib.support.check(value).all():
            return torch.tensor(float("-Inf")).to(value.device)
        else:
            raise e


def merge_dicts(dicts: List[RVDict], dim: int = 0, stack_not_cat=True) -> RVDict:
    """
    A helper function that merge multiple dicts of samples into a single dictionary,
    stacking across a new dimension
    """
    rv_keys = set().union(*(rv_dict.keys() for rv_dict in dicts))
    for idx, d in enumerate(dicts):
        if not rv_keys.issubset(d.keys()):
            raise ValueError(f"{rv_keys - d.keys()} are missing in dict {idx}")

    if stack_not_cat:
        return {rv: torch.stack([d[rv] for d in dicts], dim=dim) for rv in rv_keys}
    else:
        return {rv: torch.cat([d[rv] for d in dicts], dim=dim) for rv in rv_keys}


def seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)
