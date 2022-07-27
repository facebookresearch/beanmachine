# DO NOT REMOVE MLIRCompileError import
from beanmachine.paic2.inference.to_paic2_ast import MLIRCompileError
import unittest
from typing import Callable, List

import beanmachine.ppl as bm
import pytest
import torch
import torch.distributions as dist
from beanmachine.paic2.inference.metaworld import MetaWorld, RealWorld
from beanmachine.paic2.inference.paic2_decorators import import_inference
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world import RVDict
from torch import tensor


def fake_inference(world: MetaWorld):
    world.print()


@import_inference
def entry_point_of_fake_inference(
    queries: List[RVIdentifier],
    observations: RVDict,
    world_creator: Callable[[List[RVIdentifier], RVDict], MetaWorld],
    inference: Callable[[MetaWorld], None],
):
    inference(world_creator(queries, observations))


class SampleNormalModel:
    @bm.random_variable
    def foo(self):
        return dist.Normal(tensor(2.0), tensor(2.0))

    @bm.random_variable
    def bar(self, i):
        return dist.Normal(self.foo(), torch.tensor(1.0))


class WorldTest(unittest.TestCase):
    @pytest.mark.paic2
    def test_create_type(self):
        try:
            # create model
            model = SampleNormalModel()
            foo_value = dist.Normal(tensor(2.0), tensor(2.0)).sample(torch.Size((1, 1)))
            observations = {}
            bar_parent = dist.Normal(foo_value, torch.tensor(1.0))
            for i in range(0, 20):
                observations[model.bar(i)] = bar_parent.sample(torch.Size((1, 1)))

            # calling entry point of fake inference is logically equivalent to calling fake_inference with a world we created
            # with the same init function. Currently we just allow the init to default to random
            entry_point_of_fake_inference(
                queries=[model.foo()],
                observations=observations,
                world_creator=lambda q, o: RealWorld(q, o),
                inference=fake_inference,
            )
        except MLIRCompileError:
            self.assertFalse(True, "The python function failed to compile")


if __name__ == "__main__":
    unittest.main()
