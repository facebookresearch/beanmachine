# Copyright (c) Facebook, Inc. and its affiliates.
from collections import defaultdict

import torch
from beanmachine.ppl.world.variable import Variable


class World:
    """
    represents the world through inference run.

    takes in:
        init_world_likelihood: the likelihood of the initial world being passed
        in.
        init_world: the initial world from which the inference algorithm can
        start from. (helps us to support resumable inference)

    parameters are:
        variables_: a dict of variables keyed with their function signature.
        likelihood_: the likelihood of the world


    for instance for model below:
    class Model(StatisticalModel):
        @sample
        def foo(self):
            return dist.Bernoulli(torch.tensor(0.1))

        @sample
        def bar(self):
            if not self.foo().item():
                return dist.Bernoulli(torch.tensor(0.1))
            else:
                return dist.Bernoulli(torch.tensor(0.9))


    World.variables_ will be:

    defaultdict(<class 'beanmachine.ppl.utils.variable.Variable'>,
    {
     (<function Model.bar at 0x7ff0bb6c0b70>, ()):
        Variable(
                 distribution=Bernoulli(probs: 0.8999999761581421,
                                        logits: 2.1972243785858154),
                 value=tensor(0.),
                 parent={(<function Model.foo at 0x7ff0bb6c0a60>, ())},
                 children=set(),
                 log_prob=tensor(-2.3026)
                ),
      (<function Model.foo at 0x7ff0bb6c0a60>, ()):
         Variable(
                  distribution=Bernoulli(probs: 0.10000000149011612,
                                           logits: -2.1972246170043945),
                  value=tensor(0.),
                  parent=set(),
                  children={(<function Model.bar at 0x7ff0bb6c0b70>, ())},
                  log_prob=tensor(-0.1054)
                 )
     }
    )
    """

    def __init__(self, init_world_log_prob, init_world_dict=None):
        self.variables_ = defaultdict(Variable)
        self.log_prob_ = torch.tensor(0.0)
