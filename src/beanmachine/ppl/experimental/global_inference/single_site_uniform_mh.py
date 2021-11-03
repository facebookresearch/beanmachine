from beanmachine.ppl.experimental.global_inference.proposer.single_site_uniform_proposer import (
    SingleSiteUniformProposer,
)
from beanmachine.ppl.experimental.global_inference.single_site_inference import (
    SingleSiteInference,
)


class SingleSiteUniformMetropolisHastings(SingleSiteInference):
    def __init__(self):
        super().__init__(SingleSiteUniformProposer)
