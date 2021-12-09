# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Module defining a data extract, transform, and load API."""
from pathlib import Path
from typing import Any

import pandas as pd


class Extract:
    """Base class defining a data extraction API."""

    def extract(self) -> Any:
        """Extract data."""
        return self._extract()

    def _extract(self) -> Any:
        """Extract method to be written by the inheriting class."""
        msg = "To be implemented by the inheriting class."
        raise NotImplementedError(msg)


class Transform:
    """Base class defining a data transformation API."""

    extractor = None

    def transform(self) -> Any:
        """Transform data."""
        self.extracted_data = self.extractor().extract()
        return self._transform()

    def _transform(self) -> Any:
        """Transform method to be written by the inheriting class."""
        msg = "To be implemented by the inheriting class."
        raise NotImplementedError(msg)


class Load:
    """Base class defining a data load API."""

    transformer = None
    filename = None
    data_dir = Path(__file__).parent.joinpath("data")

    def is_cached(self) -> bool:
        return Path(self.data_dir.joinpath(self.filename)).exists()

    def load(self) -> Any:
        """Load data."""
        if self.filename is not None and self.is_cached():
            return pd.read_csv(str(self.data_dir.joinpath(self.filename)))
        self.transformed_data = self.transformer().transform()
        # Cache to disk
        if not self.data_dir.exists():
            self.data_dir.mkdir()
        self.transformed_data.to_csv(
            str(self.data_dir.joinpath(self.filename)),
            index=False,
        )
        return self._load()

    def _load(self) -> Any:
        """Load method to be written by the inheriting class."""
        msg = "To be implemented by the inheriting class."
        raise NotImplementedError(msg)
