# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Visual diagnostics tools for Bean Machine models."""

from typing import TypeVar

from beanmachine.ppl.diagnostics.tools.marginal1d.tool import Marginal1d
from beanmachine.ppl.diagnostics.tools.utils import accessor
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples


T = TypeVar("T", bound="DiagnosticsTools")


@accessor.register_mcs_accessor("diagnostics")
class DiagnosticsTools:
    """Accessor object for the visual diagnostics tools."""

    def __init__(self: T, mcs: MonteCarloSamples) -> None:
        """Initialize."""
        self.mcs = mcs
        self.idata = self.mcs.to_inference_data()

    def marginal1d(self: T) -> None:
        """Marginal 1D tool."""
        Marginal1d(self.mcs).show()
