# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Data ETL for the NBA item response tutorial."""
import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource  # usort: skip

from beanmachine.tutorials.utils import etl, plots


class ExtractNBATutorialData(etl.Extract):
    """Extraction process for the NBA data."""

    _SCHEME = "https"
    _NETLOC = "raw.githubusercontent.com"
    _PATH = (
        "polygraph-cool/last-two-minute-report/"
        + "32f1c43dfa06c2e7652cc51ea65758007f2a1a01/output"
    )
    _FILENAME = "all_games.csv"

    def __init__(self) -> None:
        self.data_url = self._build_url(self._FILENAME)
        self.extracted_data = self._extract()

    def _build_url(self, filename: str) -> str:
        return self._SCHEME + "://" + self._NETLOC + "/" + self._PATH + "/" + filename

    def _extract(self) -> pd.DataFrame:
        return pd.read_csv(self.data_url)


class TransformNBATutorialData(etl.Transform):
    """Transform NBA data for the tutorial."""

    extractor = ExtractNBATutorialData

    def _season_name(self, df: pd.DataFrame) -> None:
        # From https://en.wikipedia.org/wiki/2015%E2%80%9316_NBA_season
        season2015_16_start_date = pd.to_datetime(
            "October 27, 2015",
            format="%B %d, %Y",
        )
        season2015_16_stop_date = pd.to_datetime(
            "April 13, 2016",
            format="%B %d, %Y",
        )

        # From https://en.wikipedia.org/wiki/2016%E2%80%9317_NBA_season
        season2016_17_start_date = pd.to_datetime(
            "October 25, 2016",
            format="%B %d, %Y",
        )
        season2016_17_stop_date = pd.to_datetime(
            "April 12, 2017",
            format="%B %d, %Y",
        )

        # Create the mask and the choices.
        choices = ["2015-2016", "2016-2017"]
        conditions = [
            np.logical_and(
                df["date"] >= season2015_16_start_date,
                df["date"] <= season2015_16_stop_date,
            ),
            np.logical_and(
                df["date"] >= season2016_17_start_date,
                df["date"] <= season2016_17_stop_date,
            ),
        ]

        # Apply the mask.
        df["season"] = np.select(conditions, choices, default=None)

    def _transform(self) -> pd.DataFrame:
        # Copy the data so we can manipulate it.
        df = self.extracted_data.copy()

        # Ensure the date column is a date object.
        df["date"] = pd.to_datetime(df["date"].values, format="%Y%m%d")

        # Append the season name.
        self._season_name(df)

        # Fix spelling errors.
        teams = {
            "NKY": "NYK",
            "COS": "BOS",
            "SAT": "SAS",
            "CHi": "CHI",
            "LA)": "LAC",
            "AT)": "ATL",
            "ARL": "ATL",
        }
        columns = ["away", "home", "committing_team", "disadvantaged_team"]
        for column in columns:
            df[column] = df[column].rename(teams)

        # Fill in NaN review_decision values with INC.
        df["review_decision"] = df["review_decision"].fillna("INC")

        # Filter the data for specific foul call_types and keep only the
        # descriptors (word after the :). These types of fouls generally
        # involve two players. See
        # https://austinrochford.com/posts/2018-02-04-nba-irt-2.html for more
        # info.
        fouls = [
            "Foul: Personal",
            "Foul: Shooting",
            "Foul: Offensive",
            "Foul: Loose Ball",
            "Foul: Away from Play",
        ]
        df = df[df["call_type"].isin(fouls)]
        df["call_type"] = df["call_type"].str.split(": ", expand=True)[1].values

        # Filter the data on fourth quarters only. Then remove that column.
        df = df[df["period"] == "Q4"]
        df = df.drop("period", axis=1)

        # Only keep records that have a named season value.
        df = df.dropna(subset=["season"])

        # Remove any NaN values that may be in the players columns.
        df = df.dropna(subset=["committing_player", "disadvantaged_player"])

        # Create IDs for the players.
        committing_players = df["committing_player"].tolist()
        disadvantaged_players = df["disadvantaged_player"].tolist()
        players = sorted(set(committing_players + disadvantaged_players))
        players = {player: i for i, player in enumerate(players)}
        df["committing_player_id"] = df["committing_player"].map(players)
        df["disadvantaged_player_id"] = df["disadvantaged_player"].map(players)

        # Create IDs for the foul type.
        fouls = {name: i for i, name in enumerate(sorted(df["call_type"].unique()))}
        df["call_type_id"] = df["call_type"].map(fouls)

        # Create IDs for the season.
        seasons = {name: i for i, name in enumerate(sorted(df["season"].unique()))}
        df["season_id"] = df["season"].map(seasons)

        # New score columns.
        df["score_committing"] = (
            df["score_home"]
            .where(df["committing_team"] == df["home"], df["score_away"])
            .astype(int)
        )
        df["score_disadvantaged"] = (
            df["score_home"]
            .where(
                df["disadvantaged_team"] == df["home"],
                df["score_away"],
            )
            .astype(int)
        )

        # Round the seconds left in the game.
        df["seconds_left"] = df["seconds_left"].round(0).astype(int)

        # Foul called ID.
        df["foul_called"] = 1 * df["review_decision"].isin(["CC", "INC"])

        # Trailing flag
        df["trailing_committing"] = (
            df["score_committing"] < df["score_disadvantaged"]
        ).astype(int)

        # Calculate the difference between the teams scores.
        df["score_diff"] = df["score_disadvantaged"] - df["score_committing"]

        # Calculate the trailing possessions needed.
        df["trailing_poss"] = np.ceil(df["score_diff"].values / 3).astype(int)

        # Possessions needed ID.
        df["trailing_poss_id"] = df["trailing_poss"].map(
            {poss: i for i, poss in enumerate(sorted(df["trailing_poss"].unique()))}
        )

        # Remaining possessions.
        df["remaining_poss"] = df["seconds_left"].floordiv(25).add(1).astype(int)

        # Remaining possessions ID.
        df["remaining_poss_id"] = df["remaining_poss"].map(
            {poss: i for i, poss in enumerate(sorted(df["remaining_poss"].unique()))}
        )

        # Keep only a few columns.
        columns = [
            "seconds_left",
            "call_type",
            "call_type_id",
            "foul_called",
            "committing_player",
            "committing_player_id",
            "disadvantaged_player",
            "disadvantaged_player_id",
            "score_committing",
            "score_disadvantaged",
            "season",
            "season_id",
            "trailing_committing",
            "score_diff",
            "trailing_poss",
            "trailing_poss_id",
            "remaining_poss",
            "remaining_poss_id",
        ]
        df = df[columns]

        # Drop any duplicates.
        df = df.drop_duplicates().reset_index(drop=True)
        return df


class LoadNBATutorialData(etl.Load):
    """Load the transformed data."""

    transformer = TransformNBATutorialData
    filename = "nba.csv"

    def _load(self) -> pd.DataFrame:
        """Load transformed data."""
        return self.transformed_data


def load_data() -> pd.DataFrame:
    """Load the data."""
    loader = LoadNBATutorialData()
    return loader.load()


def plot_foul_types(series):
    tick_labels = series.index.values
    x = np.arange(len(tick_labels))
    left = x + 0.5
    y = top = series.values
    right = x - 0.5
    bottom = [0.1] * len(tick_labels)
    source = ColumnDataSource(
        {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "tick_labels": tick_labels,
            "x": x,
            "y": y,
        }
    )
    tooltips = [("Foul", "@tick_labels"), ("Count", "@top{0,0}")]
    plot = plots.bar_plot(
        plot_source=source,
        figure_kwargs={
            "title": "Foul types",
            "y_axis_label": "Counts",
            "y_axis_type": "log",
        },
        orientation="vertical",
        tooltips=tooltips,
    )
    return plot


def plot_foul_frequency(series):
    tick_labels = series.index.values
    x = np.arange(len(tick_labels))
    left = x + 0.5
    y = top = series.values
    right = x - 0.5
    bottom = np.zeros(len(tick_labels))
    source = ColumnDataSource(
        {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "tick_labels": tick_labels,
            "x": x,
            "y": y,
        }
    )
    tooltips = [("Season", "@tick_labels"), ("Frequency", "@top{0.000}")]
    plot = plots.bar_plot(
        plot_source=source,
        figure_kwargs={
            "title": "Foul frequency",
            "y_axis_label": "Frequency",
        },
        orientation="vertical",
        tooltips=tooltips,
    )
    plot.y_range.start = 0
    return plot


def plot_basic_model_residuals(residual_df):
    temp_df = residual_df.groupby("seconds_left").mean()
    plot_source = ColumnDataSource(
        {
            "x": temp_df.index.values,
            "y": temp_df["resid"].values,
        }
    )
    tooltips = [
        ("Residual", "@y{0.000}"),
        ("Seconds remaining", "@x"),
    ]

    return plots.scatter_plot(
        plot_source=plot_source,
        figure_kwargs={
            "x_axis_label": "Seconds remaining in game",
            "y_axis_label": "Residual",
            "x_range": [125, -5],
        },
        tooltips=tooltips,
    )


def plot_call_type_means(series):
    tick_labels = series.index.values
    x = np.arange(len(tick_labels))
    left = x + 0.5
    y = top = series.values
    right = x - 0.5
    bottom = np.zeros(len(tick_labels))

    plot_source = ColumnDataSource(
        {
            "x": y,
            "y": x,
            "left": bottom,
            "top": left,
            "right": top,
            "bottom": right,
            "tick_labels": tick_labels,
        }
    )
    tooltips = [
        ("Call type", "@tick_labels"),
        ("Rate", "@x{0.000}"),
    ]

    return plots.bar_plot(
        plot_source=plot_source,
        orientation="horizontal",
        figure_kwargs={
            "x_axis_label": "Observed foul call rate",
            "y_axis_label": "Call type",
        },
        tooltips=tooltips,
    )


def plot_trailing_team_committing(df):
    plot_data = (
        df.pivot_table("foul_called", "seconds_left", "trailing_committing")
        .rolling(20)
        .mean()
        .rename(columns={0: "No", 1: "Yes"})
        .rename_axis(
            "Committing team is trailing",
            axis=1,
        )
    )
    x = plot_data.index.values
    plot_sources = [
        ColumnDataSource({"x": x, "y": plot_data["No"].values}),
        ColumnDataSource({"x": x, "y": plot_data["Yes"].values}),
    ]
    labels = plot_data.columns.values
    colors = ["steelblue", "orange"]
    tooltips = [
        [("Rate", "@y{0.000}"), ("Time", "@x")],
        [("Rate", "@y{0.000}"), ("Time", "@x")],
    ]

    p = plots.line_plot(
        plot_sources,
        labels=labels,
        colors=colors,
        tooltips=tooltips,
        figure_kwargs={
            "title": "Committing team is trailing",
            "x_axis_label": "Seconds remaining",
            "y_axis_label": "Observed foul rate",
            "x_range": [125, 15],
        },
    )
    p.legend.location = "top_left"
    return p


def plot_trailing_possessions(df):
    plot_data = (
        df.pivot_table("foul_called", "seconds_left", "trailing_poss")
        .loc[:, 1:3]
        .rolling(20)
        .mean()
        .rename_axis(
            "Trailing possessions (committing team)",
            axis=1,
        )
    )
    x = plot_data.index.values
    plot_sources = [
        ColumnDataSource({"x": x, "y": plot_data[1].values}),
        ColumnDataSource({"x": x, "y": plot_data[2].values}),
        ColumnDataSource({"x": x, "y": plot_data[3].values}),
    ]
    labels = plot_data.columns.astype(str).values
    colors = ["steelblue", "orange", "brown"]
    tooltips = [
        [("Rate", "@y{0.000}"), ("Time left", "@x")],
        [("Rate", "@y{0.000}"), ("Time left", "@x")],
        [("Rate", "@y{0.000}"), ("Time left", "@x")],
    ]

    p = plots.line_plot(
        plot_sources,
        labels=labels,
        colors=colors,
        tooltips=tooltips,
        figure_kwargs={
            "title": "Trailing possessions (committing team)",
            "x_axis_label": "Seconds remaining",
            "y_axis_label": "Observed foul rate",
            "x_range": [125, 15],
        },
    )
    p.legend.location = "top_left"
    return p


def plot_possession_model_residuals(residual_df):
    temp_df = residual_df.groupby("seconds_left").mean()
    plot_source = ColumnDataSource(
        {
            "x": temp_df.index.values[::-1],
            "y": temp_df["resid"].values,
        }
    )
    tooltips = [
        ("Residual", "@y{0.000}"),
        ("Seconds remaining", "@x"),
    ]

    return plots.scatter_plot(
        plot_source=plot_source,
        figure_kwargs={
            "x_axis_label": "Seconds remaining in game",
            "y_axis_label": "Residual",
        },
        tooltips=tooltips,
    )


def plot_irt_residuals(residual_df):
    temp_df = residual_df.groupby("seconds_left").mean()
    plot_source = ColumnDataSource(
        {
            "x": temp_df.index.values[::-1],
            "y": temp_df["resid"].values,
        }
    )
    tooltips = [
        ("Residual", "@y{0.000}"),
        ("Seconds remaining", "@x"),
    ]

    return plots.scatter_plot(
        plot_source=plot_source,
        figure_kwargs={
            "x_axis_label": "Seconds remaining in game",
            "y_axis_label": "Residual",
        },
        tooltips=tooltips,
    )
