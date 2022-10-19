# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Visual diagnostics tools for Bean Machine models."""
from __future__ import annotations

from functools import wraps
from typing import Callable, TypeVar

from beanmachine.ppl.diagnostics.tools.utils import accessor
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from typing_extensions import ParamSpec

P = ParamSpec("P")
R = TypeVar("R")


def _requires_dev_packages(f: Callable[P, R]) -> Callable[P, R]:
    """A utility decorator that allow us to lazily imports the plotting modules
    and throw a useful error message when the required dependencies are not
    installed."""

    @wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return f(*args, **kwargs)
        except ModuleNotFoundError as e:
            # The diagnostic tools uses packages that are not part of the core
            # BM dependency, so we need to prompt users to manually install
            # those
            raise ModuleNotFoundError(
                "Dev packages are required for the diagnostic widgets, which "
                "can be installed with `pip install 'beanmachine[dev]'"
            ) from e

    return wrapper


@accessor.register_mcs_accessor("diagnostics")
class DiagnosticsTools:
    """Accessor object for the visual diagnostics tools."""

    def __init__(self: DiagnosticsTools, mcs: MonteCarloSamples) -> None:
        """Initialize."""
        self.mcs = mcs
        self.idata = self.mcs.to_inference_data()

    @_requires_dev_packages
    def marginal1d(self: DiagnosticsTools) -> None:
        """
        Marginal 1D diagnostic tool for a Bean Machine model.

        Returns:
            None: Displays the tool directly in a Jupyter notebook.
        """
        from beanmachine.ppl.diagnostics.tools.marginal1d.tool import Marginal1d

        Marginal1d(self.mcs).show()

    @_requires_dev_packages
    def trace(self: DiagnosticsTools) -> None:
        """
        Trace diagnostic tool for a Bean Machine model.

        Returns:
            None: Displays the tool directly in a Jupyter notebook.
        """
        from beanmachine.ppl.diagnostics.tools.trace.tool import Trace

        Trace(self.mcs).show()
