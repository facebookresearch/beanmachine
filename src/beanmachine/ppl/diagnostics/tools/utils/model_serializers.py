# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Collection of serializers for the diagnostics tool use."""

from typing import Dict, List

from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples


def serialize_bm(samples: MonteCarloSamples) -> Dict[str, List[List[float]]]:
    """
    Convert Bean Machine models to a JSON serializable object.
    Args:
        samples (MonteCarloSamples): Output of a model from Bean Machine.
    Returns
        Dict[str, List[List[float]]]: The JSON serializable object for use in the
            diagnostics tools.
    """
    rv_identifiers = list(samples.keys())
    reshaped_data = {}
    for rv_identifier in rv_identifiers:
        rv_data = samples[rv_identifier]
        rv_shape = rv_data.shape
        num_rv_chains = rv_shape[0]
        for rv_chain in range(num_rv_chains):
            chain_data = rv_data[rv_chain, :]
            chain_shape = chain_data.shape
            if len(chain_shape) > 3 and 1 not in list(chain_shape):
                msg = (
                    "Unable to handle data with dimensionality larger than " "mxnxkx1."
                )
                raise ValueError(msg)
            elif len(chain_shape) == 3 and 1 in list(chain_shape):
                if chain_shape[1] == 1 in list(chain_shape):
                    reshape_dimensions = chain_shape[2]
                else:
                    reshape_dimensions = chain_shape[1]
                for i, reshape_dimension in enumerate(range(reshape_dimensions)):
                    data = rv_data[rv_chain, :, reshape_dimension].reshape(-1)
                    reshaped_data[f"{str(rv_identifier)}[{i}]"] = data.tolist()
            elif len(chain_shape) == 1:
                reshaped_data[f"{str(rv_identifier)}"] = rv_data[rv_chain, :].tolist()
    model = dict(sorted(reshaped_data.items(), key=lambda item: item[0]))
    return model
