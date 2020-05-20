import logging

from beanmachine.ppl.diagnostics.diagnostics import Diagnostics
from beanmachine.ppl.inference import (
    CompositionalInference,
    SingleSiteAncestralMetropolisHastings,
    SingleSiteHamiltonianMonteCarlo,
    SingleSiteNewtonianMonteCarlo,
    SingleSiteRandomWalk,
    SingleSiteUniformMetropolisHastings,
)
from beanmachine.ppl.model import functional, random_variable


console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
file_handler = logging.FileHandler("beanmachine.log")
file_handler.setLevel(logging.INFO)

LOGGER = logging.getLogger("beanmachine")
LOGGER.setLevel(logging.INFO)
LOGGER.handlers.clear()
LOGGER.addHandler(console_handler)
LOGGER.addHandler(file_handler)

__all__ = [
    "CompositionalInference",
    "Diagnostics",
    "SingleSiteAncestralMetropolisHastings",
    "SingleSiteHamiltonianMonteCarlo",
    "SingleSiteNewtonianMonteCarlo",
    "SingleSiteRandomWalk",
    "SingleSiteUniformMetropolisHastings",
    "functional",
    "random_variable",
]
