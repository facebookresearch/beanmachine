# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Union

import torch
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from typing_extensions import TypeGuard


def is_rvidentifier_list(
    val: list[Union[RVIdentifier, torch.Tensor]]
) -> TypeGuard[list[RVIdentifier]]:
    """Checks whether all elements of `val` are of type RVIdentifier.

    Returns False if val is an empty list.
    """
    if val:
        return all(isinstance(x, RVIdentifier) for x in val)
    return False


def is_rvidentifier_dict(
    val: dict[Union[RVIdentifier, torch.Tensor], torch.Tensor]
) -> TypeGuard[dict[RVIdentifier, torch.Tensor]]:
    """Check whether all keys of `val` are of type RVIdentifier.

    Returns False if val is an empty dictionary.
    """
    if val:
        return all(isinstance(k, RVIdentifier) for k in val)
    return False
