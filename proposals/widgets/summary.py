"""Summary statistics widget."""
import arviz as az
import pandas as pd
from bokeh.models.sources import ColumnDataSource
from bokeh.models.widgets.tables import DataTable, TableColumn
from bokeh.plotting import show


class SummaryWidget:
    def __init__(self, idata: az.InferenceData) -> None:
        """Summary widget displaying ArviZ summary statistics.

        Args:
            idata (az.InferenceData): ArviZ `InferenceData` object.
        """
        self.idata = idata

    def modify_doc(self, doc) -> None:
        summary_df: pd.DataFrame = az.summary(self.idata, round_to=3)
        summary_df.reset_index(inplace=True)
        summary_df.rename(columns={"index": "query"}, inplace=True)
        summary_df["query"] = summary_df["query"].astype(str)
        layout = DataTable(
            columns=[
                TableColumn(field=column, title=column) for column in summary_df.columns
            ],
            source=ColumnDataSource(summary_df),
        )
        doc.add_root(layout)

    def show_widget(self):
        """Display the widget."""
        show(self.modify_doc)
