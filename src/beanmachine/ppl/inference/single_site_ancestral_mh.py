# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from beanmachine.ppl.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.inference.single_site_inference import (
    SingleSiteInference,
    JointSingleSiteInference,
)


class SingleSiteAncestralMetropolisHastings(SingleSiteInference):
    def __init__(self):
        super().__init__(SingleSiteAncestralProposer)


class GlobalAncestralMetropolisHastings(JointSingleSiteInference):
    def __init__(self):
        super().__init__(SingleSiteAncestralProposer)
