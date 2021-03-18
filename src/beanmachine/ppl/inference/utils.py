# Copyright (c) Facebook, Inc. and its affiliates
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import torch
from beanmachine.ppl.model.rv_identifier import RVIdentifier


RVDict = Dict[RVIdentifier, torch.Tensor]


class BlockType(Enum):
    """
    Enum for Block types: can be single node, or sequential block where nodes are
    sequentially re-sampled, or joint, where nodes are jointly re-sampled.
    """

    SINGLENODE = 1
    SEQUENTIAL = 2
    JOINT = 3


@dataclass(eq=True, frozen=True)
class Block:
    """
    Block class, which contains: the RVIdentifier of the first_node, type of the
    Block and list of random variables in the block in strings.
    """

    first_node: RVIdentifier
    type: BlockType
    block: List[str]


def safe_log_prob_sum(distrib, value: torch.Tensor) -> torch.Tensor:
    "Computes log_prob, converting out of support exceptions to -Infinity."
    try:
        return distrib.log_prob(value).sum()
    except (RuntimeError, ValueError) as e:
        if not distrib.support.check(value):
            return torch.tensor(float("-Inf")).to(value.device)
        else:
            raise e


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
