from beanmachine.ppl.experimental.global_inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.experimental.global_inference.single_site_inference import (
    SingleSiteInference,
    JointSingleSiteInference,
)


class SingleSiteAncestralMetropolisHastings(SingleSiteInference):
    def __init__(self):
        super().__init__(SingleSiteAncestralProposer)


class GlobalAncestralMetropolisHastings(JointSingleSiteInference):
    def __init__(self):
        super().__init__(SingleSiteAncestralProposer)
