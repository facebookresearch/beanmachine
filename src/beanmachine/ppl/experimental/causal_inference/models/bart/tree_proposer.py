# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABCMeta, abstractmethod

import torch

from .tree import Tree


class TreeProposer(metaclass=ABCMeta):
    @abstractmethod
    def propose(self, tree: Tree, X: torch.Tensor) -> Tree:
        raise NotImplementedError
