# Copyright (c) Facebook, Inc. and its affiliates.
import warnings

import beanmachine.ppl as bm
import pytest
import torch
import torch.distributions as dist


@bm.random_variable
def f():
    return 123  # BAD, needs to be a distribution


@bm.random_variable
def g(n):
    pass


@bm.functional
def h():
    return 123  # BAD; needs to be a tensor


@bm.random_variable
def flip():
    return dist.Bernoulli(0.5)


class ErrorDist(torch.distributions.Distribution):
    support = torch.distributions.constraints.real

    def __init__(self):
        self.counter = 0
        super().__init__()

    def sample(self):
        self.counter += 1
        if self.counter == 20:
            # throw error
            torch.linalg.cholesky(torch.zeros(3, 3))
        return torch.randn(1)

    def log_prob(self, *args):
        return torch.randn(1)


@bm.random_variable
def bad():
    return ErrorDist()


def test_inference_error_reporting():
    mh = bm.SingleSiteAncestralMetropolisHastings()
    with pytest.raises(TypeError) as ex:
        mh.infer(None, {}, 10)
    assert (
        str(ex.value)
        == "Parameter 'queries' is required to be a list but is of type NoneType."
    )

    with pytest.raises(TypeError) as ex:
        mh.infer([], 123, 10)
    assert (
        str(ex.value)
        == "Parameter 'observations' is required to be a dictionary but is of type int."
    )

    # Should be f():
    with pytest.raises(TypeError) as ex:
        mh.infer([f], {}, 10)
    assert (
        str(ex.value)
        == "A query is required to be a random variable but is of type function."
    )

    # Should be f():
    with pytest.raises(TypeError) as ex:
        mh.infer([f()], {f: torch.tensor(True)}, 10)
    assert (
        str(ex.value)
        == "An observation is required to be a random variable but is of type function."
    )

    # Should be a tensor
    with pytest.raises(TypeError) as ex:
        mh.infer([f()], {f(): 123.0}, 10)
    assert (
        str(ex.value)
        == "An observed value is required to be a tensor but is of type float."
    )

    # You can't make inferences on rv-of-rv
    with pytest.raises(TypeError) as ex:
        mh.infer([g(f())], {}, 10)
    assert str(ex.value) == "The arguments to a query must not be random variables."

    # You can't make inferences on rv-of-rv
    with pytest.raises(TypeError) as ex:
        mh.infer([f()], {g(f()): torch.tensor(123)}, 10)
    assert (
        str(ex.value) == "The arguments to an observation must not be random variables."
    )

    # SSAMH requires that observations must be of random variables, not
    # functionals
    with pytest.raises(TypeError) as ex:
        mh.infer([f()], {h(): torch.tensor(123)}, 10)
    assert (
        str(ex.value)
        == "An observation must observe a random_variable, not a functional."
    )

    # A functional is required to return a tensor.
    with pytest.raises(TypeError) as ex:
        mh.infer([h()], {}, 10)
    assert str(ex.value) == "The value returned by a queried function must be a tensor."

    # A random_variable is required to return a distribution
    with pytest.raises(TypeError) as ex:
        mh.infer([f()], {}, 10)
    assert str(ex.value) == "A random_variable is required to return a distribution."

    # The lookup key to the samples object is required to be an RVID.
    with pytest.raises(TypeError) as ex:
        mh.infer([flip()], {}, 10)[flip]
    assert (
        str(ex.value)
        == "The key is required to be a random variable but is of type function."
    )


def test_handle_cholesky_error():
    mh = bm.SingleSiteAncestralMetropolisHastings()
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning
        samples = mh.infer([bad()], {}, 20, num_chains=1)
        # Verify that the warning is triggered
        assert len(w) == 1
        assert "Proposal rejected" in str(w[-1])
    # Verify that the inference finishes with the right number of samples
    assert samples[bad()].shape == (1, 20, 1)
