import torch.utils._pytree as pytree
from beanmachine.ppl.experimental.global_inference.proposer.hmc_utils import (
    RealSpaceTransform,
)
from beanmachine.ppl.experimental.global_inference.simple_world import SimpleWorld
from functorch.compile import nnc_jit  # pyre-ignore[21]


# assumes that transform won't change during an inference. This is a temporary
# workaround until static_argnums in functorch is fixed
def transforms_flatten(transform: RealSpaceTransform):
    return [], {"transform": transform}


def transforms_unflatten(values, context) -> SimpleWorld:
    return context.pop("transform")


# the values of the random variable are determined by "positions" instead of values in
# world
def world_flatten(world: SimpleWorld):
    return [], {"world": world}


def world_unflatten(values, context) -> SimpleWorld:
    return context.pop("world")


pytree._register_pytree_node(
    RealSpaceTransform, transforms_flatten, transforms_unflatten
)
pytree._register_pytree_node(SimpleWorld, world_flatten, world_unflatten)


__all__ = ["nnc_jit"]
