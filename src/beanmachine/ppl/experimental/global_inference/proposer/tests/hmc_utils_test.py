import torch
from beanmachine.ppl.experimental.global_inference.proposer.hmc_utils import (
    DualAverageAdapter,
)


def test_dual_average_adapter():
    adapter = DualAverageAdapter(0.1)
    epsilon1 = adapter.step(torch.tensor(1.0))
    epsilon2 = adapter.step(torch.tensor(0.0))
    assert epsilon2 < adapter.finalize() < epsilon1
