import pytest
import torch
from beanmachine.ppl.experimental.global_inference.proposer.hmc_utils import (
    DualAverageAdapter,
    WindowScheme,
)


def test_dual_average_adapter():
    adapter = DualAverageAdapter(0.1)
    epsilon1 = adapter.step(torch.tensor(1.0))
    epsilon2 = adapter.step(torch.tensor(0.0))
    assert epsilon2 < adapter.finalize() < epsilon1


def test_small_window_scheme():
    num_adaptive_samples = 10
    scheme = WindowScheme(num_adaptive_samples)
    for _ in range(num_adaptive_samples):
        # no window should be created if num_adaptive_samples is too small
        assert not scheme.is_in_window
        scheme.step()


def test_middle_window_scheme():
    num_adaptive_samples = 125
    scheme = WindowScheme(num_adaptive_samples)
    num_windows = 0
    for i in range(num_adaptive_samples):
        if scheme.is_in_window:
            # there should be a margin at the beginning and the end of a window
            assert i > 0
            if scheme.is_end_window:
                num_windows += 1
                assert i < num_adaptive_samples
        scheme.step()
    # there should only be a single window
    assert num_windows == 1


@pytest.mark.parametrize("num_adaptive_samples", [175, 300, 399, 543])
def test_large_window_scheme(num_adaptive_samples):
    scheme = WindowScheme(num_adaptive_samples)
    window_sizes = []
    for _ in range(num_adaptive_samples):
        if scheme.is_end_window:
            window_sizes.append(scheme._window_size)
        scheme.step()
    # size of windows should be monotonically increasing
    sorted_window_sizes = sorted(window_sizes)
    assert window_sizes == sorted_window_sizes
    for win1, win2 in zip(window_sizes[:-1], window_sizes[1:-1]):
        # except for last window, window size should keep doubling
        assert win2 == win1 * 2
