# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Visual diagnostics tools for Bean Machine models."""
from typing import TypeVar

from beanmachine.ppl.diagnostics.tools.accessor import register_mcs_accessor
from beanmachine.ppl.diagnostics.tools.autocorrelation import Autocorrelation
from beanmachine.ppl.diagnostics.tools.effective_sample_size import EffectiveSampleSize
from beanmachine.ppl.diagnostics.tools.marginal1d import Marginal1d
from beanmachine.ppl.diagnostics.tools.marginal2d import Marginal2d
from beanmachine.ppl.diagnostics.tools.trace import Trace
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples


T = TypeVar("T", bound="DiagnosticsTools")


@register_mcs_accessor("diagnostics")
class DiagnosticsTools:
    """Accessor object for the visual diagnostics tools."""

    def __init__(self: T, mcs: MonteCarloSamples) -> None:
        """Initialize."""
        self.mcs = mcs
        self.idata = self.mcs.to_inference_data()

    def autocorrelation(self: T) -> None:
        """Autocorrelation tool."""
        Autocorrelation(self.idata).show_tool()

    def ess(self: T) -> None:
        """Effective Sample Size tool."""
        EffectiveSampleSize(self.idata).show_tool()

    def marginal1d(self: T) -> None:
        """Marginal 1D tool."""
        Marginal1d(self.idata).show_tool()

    def marginal2d(self: T) -> None:
        """Marginal 2D tool."""
        Marginal2d(self.idata).show_tool()

    def trace(self: T) -> None:
        """Trace tool."""
        Trace(self.idata).show_tool()

    def display_idata(self: T) -> None:
        """Display the ArviZ InferenceData object."""
        return self.idata
