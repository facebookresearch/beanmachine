# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from beanmachine.ppl.inference.proposer.single_site_uniform_proposer import (
    SingleSiteUniformProposer,
)
from beanmachine.ppl.inference.single_site_inference import (
    SingleSiteInference,
)


class SingleSiteUniformMetropolisHastings(SingleSiteInference):
    """
    Single site uniform Metropolis-Hastings. This single site algorithm proposes
    from a uniform distribution (uniform Categorical for discrete variables).
    """

    def __init__(self):
        super().__init__(SingleSiteUniformProposer)
