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
