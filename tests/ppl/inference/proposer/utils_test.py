# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from beanmachine.ppl.inference.proposer.utils import DictToVecConverter


def test_dict_to_vec_conversion():
    d = {"a": torch.ones((2, 5)), "b": torch.rand(5), "c": torch.tensor(3.0)}
    converter = DictToVecConverter(example_dict=d)
    v = converter.to_vec(d)
    assert len(v) == 16  # 2x5 + 5 + 1
    # applying exp on the flatten tensor is equivalent to applying it to each
    # of the tensor in the dictionary
    d_exp = converter.to_dict(torch.exp(v))
    for key in d:
        assert torch.allclose(torch.exp(d[key]), d_exp[key])
