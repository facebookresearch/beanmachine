import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.examples.conjugate_models.beta_binomial import BetaBinomialModel
from beanmachine.ppl.inference.single_site_ancestral_mh import (
    SingleSiteAncestralMetropolisHastings,
)
from beanmachine.ppl.inference.single_site_newtonian_monte_carlo import (
    SingleSiteNewtonianMonteCarlo,
)
import beanmachine.ppl as bm
from beanmachine.ppl.model.statistical_model import sample
from torch.autograd import grad


@bm.random_variable
def a():
    return dist.Normal(0.0, 1.0)

@bm.random_variable
def b():
    return dist.Normal(a(), 1.0)


mh = bm.SingleSiteNewtonianMonteCarlo()
mh.infer([a(), b()], {}, 10, 1)
