# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABCMeta, abstractmethod
from typing import Tuple

import torch
from beanmachine.ppl.world import World


class BaseProposer(metaclass=ABCMeta):
    @abstractmethod
    def propose(self, world: World) -> Tuple[World, torch.Tensor]:
        raise NotImplementedError

    def do_adaptation(self, world, accept_log_prob, *args, **kwargs) -> None:
        ...

    def finish_adaptation(self) -> None:
        ...
