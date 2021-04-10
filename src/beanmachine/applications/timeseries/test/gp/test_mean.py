import pytest
import torch
from sts.data import DataTensor
from sts.gp.mean import Mean


SAMPLE_DATA = [
    DataTensor(
        torch.cat([torch.randn(10, 4), torch.arange(10).unsqueeze(-1)], -1),
        header=["a", "b", "c", "d", "t"],
        normalize_cols=["t"],
    )
]


@pytest.mark.parametrize("x", SAMPLE_DATA)
def test_composition(x):
    mean = Mean(x)
    mean.add_regression(["a", "b"])
    mean.add_regression(["c"])
    assert len(mean.parts) == 3
    names = list(mean.parts)
    parts = list(mean.parts.values())
    assert names[0] == "ConstantMean"
    assert names[1] == "RegressionMean"
    # Assert that no "bias" parameter exists for regression mean
    for name, _ in parts[1].named_parameters():
        assert "bias" not in name
    assert names[2] == "RegressionMean_1"
    for name, _ in parts[2].named_parameters():
        assert "bias" not in name
    assert torch.allclose(mean(x), sum(p(x.tensor) for p in mean.parts.values()))
