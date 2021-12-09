# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from beanmachine.ppl.legacy.inference.abstract_mh_infer import AbstractMHInference
from beanmachine.ppl.legacy.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.model.rv_identifier import RVIdentifier


class SingleSiteAncestralMetropolisHastings(AbstractMHInference):
    """
    Implementation for SingleSiteAncestralMetropolisHastings
    """

    def __init__(self):
        self.proposer_ = SingleSiteAncestralProposer()
        super().__init__(proposer=self.proposer_)

    def find_best_single_site_proposer(self, node: RVIdentifier):
        """
        Finds the best proposer for a node which is
        SingleSiteAncestralMetropolisHastingsProposer for
        SingleSiteAncestralMetropolisHastings

        :param node: the node for which to return a proposer
        :returns: a proposer for the node
        """
        return self.proposer_
