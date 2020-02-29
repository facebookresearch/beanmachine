# Copyright (c) Facebook, Inc. and its affiliates
from typing import Dict, Tuple

import torch
from beanmachine.ppl.inference.proposer.newtonian_monte_carlo_utils import (
    compute_first_gradient,
    zero_grad,
)
from beanmachine.ppl.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.model.utils import RVIdentifier
from beanmachine.ppl.world import World
from torch import Tensor


class SingleSiteHamiltonianMonteCarloProposer(SingleSiteAncestralProposer):
    def __init__(self, step_size: float, num_steps: int):
        super().__init__()
        self.step_size_ = step_size
        self.num_steps_ = num_steps

    def _compute_potential_energy_gradient(self, world, node, q_unconstrained):
        score = world.propose_change_unconstrained_value(
            node, q_unconstrained, allow_graph_update=False
        )[3]
        node_var = world.get_node_in_world(node, False)
        is_valid, grad_U = compute_first_gradient(
            -score, node_var.unconstrained_value, retain_graph=True
        )

        world.reset_diff()
        return is_valid, grad_U

    def propose(self, node: RVIdentifier, world: World) -> Tuple[Tensor, Tensor, Dict]:
        """
        Proposes a new value for the node.

        :param node: the node for which we'll need to propose a new value for.
        :param world: the world in which we'll propose a new value for node.
        :returns: a new proposed value for the node and the difference in kinetic
        energy between the start and the end value
        """
        node_var = world.get_node_in_world_raise_error(node, False)
        if node_var.value is None:
            raise ValueError(f"{node} has no value")
        q_unconstrained = node_var.unconstrained_value

        # initialize momentum
        p = torch.randn(q_unconstrained.shape)
        current_K = (p * p).sum() / 2

        is_valid, grad_U = self._compute_potential_energy_gradient(
            world, node, q_unconstrained
        )
        if not is_valid:
            zero_grad(q_unconstrained)
            return super().propose(node, world)

        # take a half-step for momentum
        p = p - self.step_size_ * grad_U / 2

        # leapfrog steps
        for i in range(self.num_steps_):
            q_unconstrained = q_unconstrained.detach()
            q_unconstrained = q_unconstrained + self.step_size_ * p
            is_valid, grad_U = self._compute_potential_energy_gradient(
                world, node, q_unconstrained
            )
            if not is_valid:
                zero_grad(q_unconstrained)
                return super().propose(node, world)

            if i < self.num_steps_ - 1:
                p = p - self.step_size_ * grad_U

        # final half-step for momentum
        p = p - self.step_size_ * grad_U / 2

        zero_grad(q_unconstrained)
        # pyre-fixme
        proposed_K = (p * p).sum() / 2
        q = node_var.transform_from_unconstrained_to_constrained(q_unconstrained)
        # pyre-fixme
        return q.detach(), current_K - proposed_K, {}

    def post_process(
        self, node: RVIdentifier, world: World, auxiliary_variables: Dict
    ) -> Tensor:
        """
        Computes the log probability of going back to the old value.

        :param node: the node for which we'll need to propose a new value for.
        :param world: the world in which we have already proposed a new value
        for node.
        :returns: the log probability of proposing the old value from this new world.
        """
        return torch.tensor(0.0)
