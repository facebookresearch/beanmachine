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
    log_weight: torch.Tensor
    sum_momentums: RVDict
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
    follows Algorithm 6 of [1]. If multinomial_sampling is True, then the next state
    will be drawn from a multinomial distribution (weighted by acceptance probability,
    as introduced in Appendix 2 of [2]) instead of drawn uniformly.

    Reference:
    [1] Matthew Hoffman and Andrew Gelman. "The No-U-Turn Sampler: Adaptively
        Setting Path Lengths in Hamiltonian Monte Carlo" (2014).
        https://arxiv.org/abs/1111.4246

    [2] Michael Betancourt. "A Conceptual Introduction to Hamiltonian Monte Carlo"
        (2017). https://arxiv.org/abs/1701.02434
    """

    def __init__(
        self,
        initial_world: SimpleWorld,
        max_tree_depth: int = 10,
        max_delta_energy: float = 1000.0,
        initial_step_size: float = 1.0,
        adapt_step_size: bool = True,
        multinomial_sampling: bool = True,
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
        self._multinomial_sampling = multinomial_sampling

    def _is_u_turning(
        self,
        left_momentums: RVDict,
        right_momentums: RVDict,
        sum_momentums: RVDict,
    ) -> bool:
        """The generalized U-turn condition, as described in [2] Appendix 4.2"""
        left_angle = 0.0
        right_angle = 0.0
        for node, rho in sum_momentums.items():
            left_angle += torch.sum(left_momentums[node] * rho)
            right_angle += torch.sum(right_momentums[node] * rho)
        return bool((left_angle <= 0) or (right_angle <= 0))

    def _build_tree_base_case(self, root: _TreeNode, args: _TreeArgs) -> _Tree:
        """Base case of the recursive tree building algorithm: take a single leapfrog
        step in the specified direction and return a subtree."""
        world, momentums, pe, pe_grad = self._leapfrog_step(
            root.world, root.momentums, args.step_size * args.direction, root.pe_grad
        )
        new_energy = self._hamiltonian(world, momentums, pe)
        # initial_energy == -L(\theta^{m-1}) + 1/2 r_0^2 in Algorithm 6 of [1]
        delta_energy = torch.nan_to_num(new_energy - args.initial_energy, float("inf"))
        if self._multinomial_sampling:
            log_weight = -delta_energy
        else:
            # slice sampling as introduced in the original NUTS paper [1]
            log_weight = (args.log_slice <= -new_energy).log()

        tree_node = _TreeNode(world=world, momentums=momentums, pe_grad=pe_grad)
        return _Tree(
            left=tree_node,
            right=tree_node,
            proposal=world,
            pe=pe,
            pe_grad=pe_grad,
            log_weight=log_weight,
            sum_momentums=momentums,
            sum_accept_prob=torch.clamp(torch.exp(-delta_energy), max=1.0),
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

        # uniform progressive sampling (Appendix 3.1 of [2])
        log_weight = torch.logaddexp(sub_tree.log_weight, other_sub_tree.log_weight)
        # NaN will occur if the log weight of both subtrees are -inf
        tree_prob = torch.nan_to_num(
            torch.exp(other_sub_tree.log_weight - log_weight), 0.0
        )

        # randomly choose between left/right subtree based on their weights
        if torch.bernoulli(tree_prob):
            selected_subtree = other_sub_tree
        else:
            selected_subtree = sub_tree

        left_state = other_sub_tree.left if args.direction == -1 else sub_tree.left
        right_state = sub_tree.right if args.direction == -1 else other_sub_tree.right
        sum_momentums = {
            node: sub_tree.sum_momentums[node] + other_sub_tree.sum_momentums[node]
            for node in sub_tree.sum_momentums
        }
        return _Tree(
            left=left_state,
            right=right_state,
            proposal=selected_subtree.proposal,
            pe=selected_subtree.pe,
            pe_grad=selected_subtree.pe_grad,
            log_weight=log_weight,
            sum_momentums=sum_momentums,
            sum_accept_prob=sub_tree.sum_accept_prob + other_sub_tree.sum_accept_prob,
            num_proposals=sub_tree.num_proposals + other_sub_tree.num_proposals,
            turned_or_diverged=other_sub_tree.turned_or_diverged
            or self._is_u_turning(
                left_state.momentums,
                right_state.momentums,
                sum_momentums,
            ),
        )

    def propose(self, world: Optional[SimpleWorld] = None) -> SimpleWorld:
        if world is not None and world is not self.world:
            # re-compute cached values in case world was modified by other sources
            self._pe, self._pe_grad = self._potential_grads(self.world)
            self.world = world

        momentums = self._initialize_momentums(self.world)
        current_energy = self._hamiltonian(self.world, momentums, self._pe)
        if self._multinomial_sampling:
            # log slice is only used to check the divergence
            log_slice = -current_energy
        else:
            # this is a more stable way to sample from log(Uniform(0, exp(-current_energy)))
            log_slice = torch.log1p(-torch.rand(())) - current_energy
        left_tree_node = right_tree_node = _TreeNode(
            self.world, momentums, self._pe_grad
        )
        log_weight = torch.tensor(0.0)  # log accept prob of staying at current state
        sum_accept_prob = 0.0
        num_proposals = 0
        sum_momentums = momentums

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

            # biased progressive sampling (Appendix 3.2 of [2])
            tree_prob = torch.clamp(torch.exp(tree.log_weight - log_weight), max=1.0)

            # choose new world by randomly sample from proposed worlds
            if torch.bernoulli(tree_prob):
                self.world, self._pe, self._pe_grad = (
                    tree.proposal,
                    tree.pe,
                    tree.pe_grad,
                )
            sum_momentums = {
                node: sum_momentums[node] + tree.sum_momentums[node]
                for node in sum_momentums
            }
            if self._is_u_turning(
                left_tree_node.momentums,
                right_tree_node.momentums,
                sum_momentums,
            ):
                break

            log_weight = torch.logaddexp(log_weight, tree.log_weight)

        self._alpha = sum_accept_prob / num_proposals
        return self.world
