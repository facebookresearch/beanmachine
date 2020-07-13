# Copyright (c) Facebook, Inc. and its affiliates.
from torch import Tensor


def ensure_1d(t: Tensor) -> Tensor:
    ndim = t.dim()
    if ndim > 1:
        raise NotImplementedError(
            f"IC currently only supports 0D (scalar) and 1D (vector) values. "
            f"Encountered tensor={t} with dim={ndim}"
        )
    if ndim == 1:
        return t
    else:
        return t.unsqueeze(0)
