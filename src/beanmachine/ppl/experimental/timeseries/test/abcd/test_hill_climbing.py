# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import pandas as pd
import pytest
import torch
from sts.abcd.hill_climbing import HillClimbingSearch
from sts.abcd.utils import GRAMMAR_RULES
from sts.data import df_to_tensor


@pytest.mark.skip(reason="unable to run multiprocessing")
def teset_hill_climbing_parallel():
    num_data = 20
    x_orig = torch.linspace(0, 1, num_data)
    y_orig = x_orig * 8 + 2.0 + 0.2 * torch.randn(x_orig.shape)
    train_num = round(0.8 * num_data)
    data = {"x": x_orig, "y": y_orig}
    df = pd.DataFrame(data, dtype=float)
    data = df_to_tensor(df, normalize_cols=True)
    x_train = data[:train_num, ["x"]]
    y_train = data[:train_num, "y"]

    rules = GRAMMAR_RULES["basic"]
    searcher = HillClimbingSearch(x_train.tensor, y_train.tensor)
    searcher.search(
        num_restarts=2,
        num_iters=5,
        top_k=2,
        output_file="./search_airline_hillclimb_nonparallel_pkl.txt",
        pickle_file="./search_airline_hillclimb_nonparallel.pkl",
        run_in_parallel=True,
        grammar=rules,
    )
