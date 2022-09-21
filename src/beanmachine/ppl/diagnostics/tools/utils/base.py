# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Base class for diagnostic tools of a Bean Machine model."""

import re
from abc import ABC, abstractmethod
from typing import TypeVar

from beanmachine.ppl.diagnostics.tools import JS_DIST_DIR
from beanmachine.ppl.diagnostics.tools.utils import plotting_utils
from beanmachine.ppl.diagnostics.tools.utils.model_serializers import serialize_bm
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from IPython.display import display, HTML


T = TypeVar("T", bound="Base")


class Base(ABC):
    @abstractmethod
    def __init__(self: T, mcs: MonteCarloSamples) -> None:
        self.data = serialize_bm(mcs)
        self.rv_names = list(self.data.keys())
        self.num_chains = mcs.num_chains
        self.num_draws = mcs.get_num_samples()
        self.palette = plotting_utils.choose_palette(self.num_chains)
        self.js = self.load_js()

    def load_js(self: T) -> str:
        name = self.__class__.__name__
        name_tokens = re.findall(r"[A-Z][^A-Z]*", name)
        name = "_".join(name_tokens)
        path = JS_DIST_DIR.joinpath(f"{name.lower()}.js")
        with path.open() as f:
            js = f.read()
        return js

    def show(self: T):
        display(HTML(self.create_document()))

    @abstractmethod
    def create_document(self: T) -> str:
        ...
