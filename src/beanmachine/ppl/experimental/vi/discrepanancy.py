# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"Csiszar f-functions in log-space."

import torch


def kl_reverse(logu: torch.Tensor) -> torch.Tensor:
    """
    Log-space Csiszar function for reverse KL-divergence D_f(p,q) = KL(q||p).

    Also known as the exclusive KL-divergence and negative ELBO, minimizing
    results in zero-forcing / mode-seeking behavior.

    Args:
        logu (torch.Tensor): ``p.log_prob``s evaluated at samples from q.
    """
    return -logu


def kl_forward(logu: torch.Tensor) -> torch.Tensor:
    """
    Log-space Csiszar function for forward KL-divergence D_f(p,q) = KL(p||q).

    Also known as the inclusive KL-divergence, minimizing results in
    zero-avoiding / mass-covering behavior.

    Args:
        logu (torch.Tensor): ``p.log_prob``s evaluated at samples from q.
    """
    return torch.exp(logu) * logu
