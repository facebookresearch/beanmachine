# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Generic, TypeVar

import torch

KeyType = TypeVar("KeyType")


class DictToVecConverter(Generic[KeyType]):
    """
    A utility class to convert a dictionary of Tensors into a single flattened
    Tensor or the other way around.

    Args:
        example_dict: A dict that will be used to determine the order of the
        keys and the size of the flattened Tensor.
    """

    def __init__(self, example_dict: Dict[KeyType, torch.Tensor]) -> None:
        # determine the order of the keys
        self._keys = list(example_dict.keys())
        # store the size of the values, which will be used when we want to
        # reshape them back
        self._val_shapes = [example_dict[key].shape for key in self._keys]
        # compute the indicies that each of the entry corresponds to. e.g. for
        # keys[0], its value will correspond to flatten_vec[idxs[0] : idxs[1]]
        val_sizes = [example_dict[key].numel() for key in self._keys]
        self._idxs = list(torch.cumsum(torch.tensor([0] + val_sizes), dim=0))

    def to_vec(self, dict_in: Dict[KeyType, torch.Tensor]) -> torch.Tensor:
        """Concatenate the entries of a dictionary to a flattened Tensor"""
        return torch.cat([dict_in[key].flatten() for key in self._keys])

    def to_dict(self, vec_in: torch.Tensor) -> Dict[KeyType, torch.Tensor]:
        """Reconstruct a dictionary out of a flattened Tensor"""
        retval = {}
        for key, shape, idx_begin, idx_end in zip(
            self._keys, self._val_shapes, self._idxs, self._idxs[1:]
        ):
            retval[key] = vec_in[idx_begin:idx_end].reshape(shape)
        return retval
