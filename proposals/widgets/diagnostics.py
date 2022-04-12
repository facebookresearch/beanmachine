"""Diagnostic module for rendering model diagnostic widgets in a Jupyter notebook."""
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples

from .accessor import register_mcs_accessor
from .autocorrelation import AutocorrelationWidget
from .effective_sample_size import EffectiveSampleSizeWidget
from .joint_plot import JointPlotWidget
from .posterior import PosteriorWidget
from .summary import SummaryWidget
from .trace_plot import TracePlotWidget


@register_mcs_accessor("diagnostics")
class DiagnosticsWidgets:
    def __init__(self, mcs: MonteCarloSamples) -> None:
        self._mcs = mcs
        self._idata = self._mcs.to_inference_data()

    def autocorrelation(self):
        AutocorrelationWidget(self._idata).show_widget()

    def ess(self):
        EffectiveSampleSizeWidget(self._idata).show_widget()

    def joint_plot(self):
        JointPlotWidget(self._idata).show_widget()

    def posterior(self):
        PosteriorWidget(self._idata).show_widget()

    def summary(self):
        SummaryWidget(self._idata).show_widget()

    def trace_plot(self):
        TracePlotWidget(self._idata).show_widget()
