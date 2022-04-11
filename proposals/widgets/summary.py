"""Summary statistics widget."""
import arviz as az
import pandas as pd
from bokeh.models.sources import ColumnDataSource
from bokeh.models.widgets.markups import Div
from bokeh.models.widgets.panels import Panel, Tabs
from bokeh.models.widgets.tables import DataTable, TableColumn
from bokeh.plotting import show


class SummaryWidget:
    def __init__(self, idata: az.InferenceData) -> None:
        """Summary widget displaying ArviZ summary statistics.

        Args:
            idata (az.InferenceData): ArviZ `InferenceData` object.
        """
        self.idata = idata

    def help_page(self):
        text = """
            <h2>
              Measuring variance between- and within-chains with \(\hat{R}\)
            </h2>
            <p style="margin-bottom: 10px">
              \(\hat{R}\) is a diagnostic tool that measures the between- and
              within-chain variances. It is a test that indicates a lack of convergence
              by comparing the variance between multiple chains to the variance within
              each chain. If the parameters are successfully exploring the full space
              for each chain, then \(\hat{R}\\approx1\), since the between-chain and
              within-chain variance should be equal. \(\hat{R}\) is calculated from
              \(N\) samples as
            </p>
            <p style="margin-bottom: 10px; text-align: center;">
                \(\\begin{aligned}
                \hat{R} &= \\frac{\hat{V}}{W} \\\\
                \hat{V} &= \\frac{N-1}{N} W + \\frac{1}{N} B,
                \end{aligned}\)
            </p>
            <p style="margin-bottom: 10px">
              where \(W\) is the within-chain variance, \(B\) is the between-chain
              variance and \(\hat{V}\) is the estimate of the posterior variance of the
              samples. The take-away here is that \(\hat{R}\) converges to 1 when each
              of the chains begins to empirically approximate the same posterior
              distribution. We do not recommend using inference results if
              \(\hat{R}>1.01\). More information about \(\hat{R}\) can be found in the
              Vehtari <em>et al</em> paper linked below.
            </p>
            <h2>
              Effective sample size diagnostic
            </h2>
            <p style="margin-bottom: 10px">
              MCMC samplers do not draw truly independent samples from the target
              distribution, which means that our samples are correlated. In an ideal
              situation all samples would be independent, but we do not have that
              luxury. We can, however, measure the number of <em>effectively
              independent</em> samples we draw, which is called the effective sample
              size. You can read more about how this value is calculated in the Vehtari
              <em>et al</em> paper. In brief, it is a measure that combines information
              from the \(\hat{R}\) value with the autocorrelation estimates within the
              chains.
            </p>
            <p style="margin-bottom: 10px">
              ESS estimates come in two variants, <em>ess_bulk</em> and
              <em>ess_tail</em>. The former is the default, but the latter can be useful
              if you need good estimates of the tails of your posterior distribution.
              The rule of thumb for <em>ess_bulk</em> is for this value to be greater
              than 100 per chain on average. The <em>ess_tail</em> is an estimate for
              effectively independent samples considering the more extreme values of the
              posterior. This is not the number of samples that landed in the tails of
              the posterior, but rather a measure of the number of effectively
              independent samples if we sampled the tails of the posterior. The rule of
              thumb for this value is also to be greater than 100 per chain on average.
            </p>
            <p style="margin-bottom: 10px">
              When the model is converging properly, both the bulk and tail lines should
              be <em>roughly</em> linear.
            </p>
            <ul>
              <li>
                Vehtari A, Gelman A, Simpson D, Carpenter B, Bürkner PC (2021)
                <b>
                  Rank-normalization, folding, and localization: An improved \(\hat{R}\)
                  for assessing convergence of MCMC (with discussion)
                </b>.
                <em>Bayesian Analysis</em> 16(2)
                667–718.
                <a href=https://dx.doi.org/10.1214/20-BA1221 style="color: blue">
                  doi: 10.1214/20-BA1221
                </a>.
              </li>
            </ul>
        """
        div = Div(text=text, disable_math=False, min_width=800)
        return div

    def modify_doc(self, doc) -> None:

        """Modify the document by adding the widget."""
        summary_df: pd.DataFrame = az.summary(self.idata, round_to=3)
        summary_df.reset_index(inplace=True)
        summary_df.rename(columns={"index": "query"}, inplace=True)
        summary_df["query"] = summary_df["query"].astype(str)
        summary_table = DataTable(
            columns=[
                TableColumn(field=column, title=column) for column in summary_df.columns
            ],
            source=ColumnDataSource(summary_df),
        )
        summary_panel = Panel(child=summary_table, title="Summary")
        help_panel = Panel(child=self.help_page(), title="Help")
        tabs = Tabs(tabs=[summary_panel, help_panel])
        layout = tabs
        doc.add_root(layout)

    def show_widget(self):
        """Display the widget."""
        show(self.modify_doc)
