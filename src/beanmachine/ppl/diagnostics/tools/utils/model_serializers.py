# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Collection of serializers for the diagnostics tool use."""

from typing import Dict, List

from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples


def serialize_bm(samples: MonteCarloSamples) -> Dict[str, List[List[float]]]:
    """Convert Bean Machine models to a JSON serializable object.

    Parameters
    ----------
    samples : MonteCarloSamples
        Output of a model from Bean Machine.

    Returns
    -------
    model : Dict[str, List[List[float]]]
        The JSON serializable object for use in the diagnostics tools.
    """
    model = dict(
        sorted(
            {str(key): value.tolist() for key, value in samples.items()}.items(),
            key=lambda item: item[0],
        ),
    )
    return model
