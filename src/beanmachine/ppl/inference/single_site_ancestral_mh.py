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
