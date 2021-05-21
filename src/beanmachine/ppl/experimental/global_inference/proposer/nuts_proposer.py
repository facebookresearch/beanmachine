from typing import NamedTuple, Optional

import torch
from beanmachine.ppl.experimental.global_inference.proposer.hmc_proposer import (
    HMCProposer,
)
from beanmachine.ppl.experimental.global_inference.simple_world import (
    RVDict,
    SimpleWorld,
)


class _TreeNode(NamedTuple):
    world: SimpleWorld
    momentums: RVDict
    pe_grad: RVDict


class _Tree(NamedTuple):
    left: _TreeNode
    right: _TreeNode
    proposal: SimpleWorld
    pe: torch.Tensor
    pe_grad: RVDict
    weight: torch.Tensor
    sum_accept_prob: torch.Tensor
    num_proposals: int
    turned_or_diverged: bool


class _TreeArgs(NamedTuple):
    log_slice: torch.Tensor
    direction: int
    step_size: float
    initial_energy: torch.Tensor


class NUTSProposer(HMCProposer):
    """
    The No-U-Turn Sampler (NUTS) as described in [1]. Unlike vanilla HMC, it does not
    require users to specify a trajectory length. The current implementation roughly
    follows Algorithm 6 of [1].

    Reference:
    [1] Matthew Hoffman and Andrew Gelman. "The No-U-Turn Sampler: Adaptively
        Setting Path Lengths in Hamiltonian Monte Carlo" (2014).
        https://arxiv.org/abs/1111.4246
    """

    def __init__(
        self,
        initial_world: SimpleWorld,
        max_tree_depth: int = 10,
        max_delta_energy: float = 1000.0,
        initial_step_size: float = 1.0,
        adapt_step_size: bool = True,
    ):
        # note that trajectory_length is not used in NUTS
        super().__init__(
            initial_world,
            trajectory_length=0.0,
            initial_step_size=initial_step_size,
            adapt_step_size=adapt_step_size,
        )
        self._max_tree_depth = max_tree_depth
        self._max_delta_energy = max_delta_energy

    def _is_u_turning(self, left_state: _TreeNode, right_state: _TreeNode) -> bool:
        left_angle = 0.0
        right_angle = 0.0
        for node in left_state.world.latent_nodes:
            diff = right_state.world[node] - left_state.world[node]
            left_angle += torch.sum(diff * left_state.momentums[node])
            right_angle += torch.sum(diff * right_state.momentums[node])
        return bool((left_angle <= 0) or (right_angle <= 0))

    def _build_tree_base_case(self, root: _TreeNode, args: _TreeArgs) -> _Tree:
        """Base case of the recursive tree building algorithm: take a single leapfrog
        step in the specified direction and return a subtree."""
        world, momentums, pe, pe_grad = self._leapfrog_step(
            root.world, root.momentums, args.step_size * args.direction, root.pe_grad
        )
        new_energy = self._hamiltonian(world, momentums, pe)
        new_energy = torch.nan_to_num(new_energy, float("inf"))
        tree_node = _TreeNode(world=world, momentums=momentums, pe_grad=pe_grad)
        return _Tree(
            left=tree_node,
            right=tree_node,
            proposal=world,
            pe=pe,
            pe_grad=pe_grad,
            weight=(args.log_slice <= -new_energy).float(),
            sum_accept_prob=torch.clamp(
                # initial_energy == -L(\theta^{m-1}) + 1/2 r_0^2 in Algorithm 6 of [1]
                torch.exp(args.initial_energy - new_energy),
                max=1.0,
            ),
            num_proposals=1,
            turned_or_diverged=bool(
                args.log_slice >= self._max_delta_energy - new_energy
            ),
        )

    def _build_tree(self, root: _TreeNode, tree_depth: int, args: _TreeArgs) -> _Tree:
        """Build the binary tree by recursively build the left and right subtrees and
        combine the two."""
        if tree_depth == 0:
            return self._build_tree_base_case(root, args)

        # build the first half of the tree
        sub_tree = self._build_tree(root, tree_depth - 1, args)
        if sub_tree.turned_or_diverged:
            return sub_tree

        # build the other half of the tree
        other_sub_tree = self._build_tree(
            root=sub_tree.left if args.direction == -1 else sub_tree.right,
            tree_depth=tree_depth - 1,
            args=args,
        )

        # randomly choose between left/right subtree based on their weights
        sum_weight = sub_tree.weight + other_sub_tree.weight
        # clamp with a non-zero minimum value to avoid divide-by-zero
        if torch.bernoulli(other_sub_tree.weight / torch.clamp(sum_weight, min=1e-3)):
            selected_subtree = other_sub_tree
        else:
            selected_subtree = sub_tree

        left_state = other_sub_tree.left if args.direction == -1 else sub_tree.left
        right_state = sub_tree.right if args.direction == -1 else other_sub_tree.right
        return _Tree(
            left=left_state,
            right=right_state,
            proposal=selected_subtree.proposal,
            pe=selected_subtree.pe,
            pe_grad=selected_subtree.pe_grad,
            weight=sum_weight,
            sum_accept_prob=sub_tree.sum_accept_prob + other_sub_tree.sum_accept_prob,
            num_proposals=sub_tree.num_proposals + other_sub_tree.num_proposals,
            turned_or_diverged=other_sub_tree.turned_or_diverged
            or self._is_u_turning(left_state, right_state),
        )

    def propose(self, world: Optional[SimpleWorld] = None) -> SimpleWorld:
        if world is not None and world is not self.world:
            # re-compute cached values in case world was modified by other sources
            self._pe, self._pe_grad = self._potential_grads(self.world)
            self.world = world

        momentums = self._initialize_momentums(self.world)
        current_energy = self._hamiltonian(self.world, momentums, self._pe)
        # this is a more stable way to sample from log(Uniform(0, exp(-current_energy)))
        log_slice = torch.log1p(-torch.rand(())) - current_energy
        left_tree_node = right_tree_node = _TreeNode(
            self.world, momentums, self._pe_grad
        )
        sum_weight = 1.0
        sum_accept_prob = 0.0
        num_proposals = 0

        for j in range(self._max_tree_depth):
            direction = 1 if torch.rand(()) > 0.5 else -1
            tree_args = _TreeArgs(log_slice, direction, self.step_size, current_energy)
            if direction == -1:
                tree = self._build_tree(left_tree_node, j, tree_args)
                left_tree_node = tree.left
            else:
                tree = self._build_tree(right_tree_node, j, tree_args)
                right_tree_node = tree.right

            sum_accept_prob += tree.sum_accept_prob
            num_proposals += tree.num_proposals

            if tree.turned_or_diverged:
                break

            # choose new world by randomly sample from proposed worlds
            if torch.bernoulli(torch.clamp(tree.weight / sum_weight, max=1.0)):
                self.world, self._pe, self._pe_grad = (
                    tree.proposal,
                    tree.pe,
                    tree.pe_grad,
                )

            if self._is_u_turning(left_tree_node, right_tree_node):
                break

            sum_weight += tree.weight

        self._alpha = sum_accept_prob / num_proposals
        return self.world
