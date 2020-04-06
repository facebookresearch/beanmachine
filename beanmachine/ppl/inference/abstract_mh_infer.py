# Copyright (c) Facebook, Inc. and its affiliates.
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from random import shuffle
from typing import Dict, List, Tuple

import torch
import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.inference.abstract_infer import AbstractInference
from beanmachine.ppl.inference.utils import Block, BlockType
from beanmachine.ppl.model.statistical_model import StatisticalModel
from beanmachine.ppl.model.utils import Mode, RVIdentifier
from torch import Tensor
from tqdm import tqdm


class AbstractMHInference(AbstractInference, metaclass=ABCMeta):
    """
    Abstract inference object that all single-site MH inference algorithms
    inherit from.
    """

    def __init__(self):
        super().__init__()
        self.blocks_ = []

    def initialize_world(self):
        """
        Initializes the world variables with queries and observation calls.

        :param queries: random variables to query
        :param observations: observed random variables with their values
        """
        self.world_.set_observations(self.observations_)
        StatisticalModel.set_mode(Mode.INFERENCE)
        for node in self.observations_:
            # makes the call for the observation node, which will run sample(node())
            # that results in adding its corresponding Variable and its dependent
            # Variable to the world
            node.function._wrapper(*node.arguments)
        for node in self.queries_:
            # makes the call for the query node, which will run sample(node())
            # that results in adding its corresponding Variable and its dependent
            # Variable to the world.
            node.function._wrapper(*node.arguments)

        self.world_.accept_diff()

    def accept_or_reject_update(
        self,
        node_log_update: Tensor,
        children_log_updates: Tensor,
        proposal_log_update: Tensor,
    ) -> Tuple[bool, Tensor]:
        """
        Accepts or rejects the change in the diff by setting a stochastic
        threshold by drawing a sample from a Uniform distribution. It accepts
        the change if sum of all log_prob updates are larger than this threshold
        and rejects otherwise.

        :param node_log_update: log_prob update to the node that was resampled
        from.
        :param children_log_updates: log_prob updates of the immediate children
        of the node that was resampled from.
        :param proposal_log_update: log_prob update of the proposal
        :returns: acceptance probability of proposal
        """
        log_update = children_log_updates + node_log_update + proposal_log_update

        is_accepted = False
        if log_update >= tensor(0.0):
            self.world_.accept_diff()
            is_accepted = True
        else:
            alpha = dist.Uniform(tensor(0.0), tensor(1.0)).sample().log()
            if log_update > alpha:
                self.world_.accept_diff()
                is_accepted = True
            else:
                self.world_.reject_diff()
                is_accepted = False
        acceptance_prob = torch.min(
            tensor(1.0, dtype=log_update.dtype), torch.exp(log_update)
        )
        return is_accepted, acceptance_prob

    def single_inference_run(self, node: RVIdentifier, proposer) -> Tuple[bool, Tensor]:
        """
        Run one iteration of the inference algorithms for a given node which is
        to follow the steps below:
        1) Propose a new value for the node
        2) Update the world given the new value
        3) Compute the log proposal ratio of proposing this value
        4) Accept or reject the proposed value

        :param node: the node to be re-sampled in this inference run
        :param proposer: the proposer with which propose a new value for node
        :returns: acceptance probability for the query
        """
        proposed_value, negative_proposal_log_update, auxiliary_variables = proposer.propose(
            node, self.world_
        )

        children_log_updates, _, node_log_update, _ = self.world_.propose_change(
            node, proposed_value
        )
        positive_proposal_log_update = proposer.post_process(
            node, self.world_, auxiliary_variables
        )
        proposal_log_update = (
            positive_proposal_log_update + negative_proposal_log_update
        )
        is_accepted, acceptance_probability = self.accept_or_reject_update(
            node_log_update, children_log_updates, proposal_log_update
        )
        return is_accepted, acceptance_probability

    def block_propose_change(self, block: Block) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Propose values for a block of random variable

        :param block: the block to propose new value for. A block is a group of
        random variable which we will sequentially update and accept their
        values all-together.
        :param world: the world in which a new value for block is going to be
        proposed.

        :returns: nodes_log_updates, children_log_updates and
        proposal_log_updates of the values proposed for the block.
        """
        markov_blanket = set({block.first_node})
        markov_blanket_func = {}
        markov_blanket_func[block.first_node.function._wrapper] = [block.first_node]
        pos_proposal_log_updates, neg_proposal_log_updates = tensor(0.0), tensor(0.0)
        children_log_updates, nodes_log_updates = tensor(0.0), tensor(0.0)
        # We will go through all family of random variable in the block. Note
        # that in block we have family of X and not the specific random variable
        # X(1)
        for node_func in block.block:
            # We then look up which of the random variable in the family are in
            # the markov blanket
            if node_func not in markov_blanket_func:
                continue
            # We will go through all random variables that are both in the
            # markov blanket and block.
            for node in markov_blanket_func[node_func].copy():
                # We look up the node's current markov blanket before re-sampling
                old_node_markov_blanket = self.world_.get_markov_blanket(node)
                proposer = self.find_best_single_site_proposer(node)
                # We use the best single site proposer to propose a new value.
                proposed_value, negative_proposal_log_update, auxiliary_variables = proposer.propose(
                    node, self.world_
                )
                neg_proposal_log_updates += negative_proposal_log_update

                # We update the world (through a new diff in the diff stack).
                children_log_update, _, node_log_update, _ = self.world_.propose_change(
                    node, proposed_value, start_new_diff=True
                )
                children_log_updates += children_log_update
                nodes_log_updates += node_log_update
                pos_proposal_log_updates += proposer.post_process(
                    node, self.world_, auxiliary_variables
                )
                # We look up the updated markov blanket of the re-sampled node.
                new_node_markov_blanket = self.world_.get_markov_blanket(node)
                all_node_markov_blanket = (
                    old_node_markov_blanket | new_node_markov_blanket
                )
                # new_nodes_to_be_added is all the new nodes to be added to
                # entire markov blanket.
                new_nodes_to_be_added = all_node_markov_blanket - markov_blanket
                for new_node in new_nodes_to_be_added:
                    if new_node is None:
                        continue
                    # We create a dictionary from node family to the node itself
                    # as the match with block happens at the family level and
                    # this makes the lookup much faster.
                    if new_node.function._wrapper not in markov_blanket_func:
                        markov_blanket_func[new_node.function._wrapper] = []
                    markov_blanket_func[new_node.function._wrapper].append(new_node)
                markov_blanket |= new_nodes_to_be_added

        proposal_log_updates = pos_proposal_log_updates + neg_proposal_log_updates
        return nodes_log_updates, children_log_updates, proposal_log_updates

    def single_inference_run_with_sequential_block_update(self, block: Block):
        """
        Run one iteration of the inference algorithm for a given block.

        :param block: the block of random variables to be resampled sequentially
        in this inference run
        """
        nodes_log_updates, children_log_updates, proposal_log_updates = self.block_propose_change(
            block
        )
        self.accept_or_reject_update(
            nodes_log_updates, children_log_updates, proposal_log_updates
        )

    def process_blocks(self) -> List[Block]:
        """
        Process all blocks.

        :returns: list of blocks in Block class which includes all variables in
        the world as well as blocks passed in the by the user
        """
        blocks = []
        for node in self.world_.get_all_world_vars():
            if node in self.observations_:
                continue
            blocks.append(Block(first_node=node, type=BlockType.SINGLENODE, block=[]))
        for block in self.blocks_:
            first_node_str = block[0]
            first_nodes = self.world_.get_all_nodes_from_func(first_node_str)
            for node in first_nodes:
                blocks.append(
                    Block(first_node=node, type=BlockType.SEQUENTIAL, block=block)
                )

        return blocks

    @abstractmethod
    def find_best_single_site_proposer(self, node: RVIdentifier):
        """
        Finds the best proposer for a node.

        :param node: the node for which to return a proposer
        :returns: a proposer for the node
        """
        raise NotImplementedError(
            "Inference algorithm must implement find_best_proposer."
        )

    def _infer(
        self, num_samples: int, num_adapt_steps: int = 0
    ) -> Dict[RVIdentifier, Tensor]:
        """
        Run inference algorithms.

        :param num_samples: number of samples to collect for the query.
        :param num_adapt_steps: number of steps to adapt/tune the proposer.
        :returns: samples for the query
        """
        self.initialize_world()
        queries_sample = defaultdict()

        for iteration in tqdm(iterable=range(num_samples), desc="Samples collected"):
            blocks = self.process_blocks()
            shuffle(blocks)
            for block in blocks:
                if block.type == BlockType.SINGLENODE:
                    node = block.first_node
                    if node in self.observations_:
                        continue
                    if not self.world_.contains_in_world(node):
                        continue

                    proposer = self.find_best_single_site_proposer(node)
                    is_accepted, acceptance_probability = self.single_inference_run(
                        node, proposer
                    )
                    if iteration < num_adapt_steps:
                        proposer.do_adaptation(
                            node,
                            self.world_,
                            acceptance_probability,
                            iteration,
                            num_adapt_steps,
                            is_accepted,
                        )

                if block.type == BlockType.SEQUENTIAL and iteration >= num_adapt_steps:
                    self.single_inference_run_with_sequential_block_update(block)

            for query in self.queries_:
                # unsqueeze the sampled value tensor, which adds an extra dimension
                # along which we'll be adding samples generated at each iteration
                if query not in queries_sample:
                    queries_sample[query] = (
                        query.function._wrapper(*query.arguments).unsqueeze(0).clone()
                    )
                else:
                    queries_sample[query] = torch.cat(
                        [
                            queries_sample[query],
                            query.function._wrapper(*query.arguments)
                            .unsqueeze(0)
                            .clone(),
                        ],
                        dim=0,
                    )
            self.world_.accept_diff()
        return queries_sample
