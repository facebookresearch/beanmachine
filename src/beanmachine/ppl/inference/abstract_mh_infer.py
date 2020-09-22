# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from random import shuffle
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.inference.abstract_infer import AbstractInference, VerboseLevel
from beanmachine.ppl.inference.utils import Block, BlockType
from beanmachine.ppl.model.statistical_model import StatisticalModel
from beanmachine.ppl.model.utils import LogLevel, Mode, RVIdentifier, get_wrapper
from beanmachine.ppl.world.variable import TransformType
from torch import Tensor
from tqdm.auto import tqdm  # pyre-ignore


LOGGER_INFERENCE = logging.getLogger("beanmachine.inference")
LOGGER_PROPOSER = logging.getLogger("beanmachine.proposer")
LOGGER_WORLD = logging.getLogger("beanmachine.world")


class AbstractMHInference(AbstractInference, metaclass=ABCMeta):
    """
    Abstract inference object that all single-site MH inference algorithms
    inherit from.
    """

    def __init__(
        self,
        proposer=None,
        transform_type: TransformType = TransformType.NONE,
        transforms: Optional[List] = None,
        skip_single_inference_run: bool = False,
    ):
        super().__init__()
        self.world_.set_all_nodes_proposer(proposer)
        self.world_.set_all_nodes_transform(transform_type, transforms)
        self.blocks_ = []
        self.skip_single_inference_run = skip_single_inference_run

    def initialize_world(self, initialize_from_prior: bool = False):
        """
        Initializes the world variables with queries and observation calls.

        :param initialize_from_prior: boolean to initialize samples from prior
        """
        self.initialize_infer()
        self.world_.set_observations(self.observations_)
        self.world_.set_initialize_from_prior(initialize_from_prior)
        StatisticalModel.set_mode(Mode.INFERENCE)
        for node in self.observations_:
            # makes the call for the observation node, which will run sample(node())
            # that results in adding its corresponding Variable and its dependent
            # Variable to the world
            get_wrapper(node.function)(*node.arguments)
        for node in self.queries_:
            # makes the call for the query node, which will run sample(node())
            # that results in adding its corresponding Variable and its dependent
            # Variable to the world.
            get_wrapper(node.function)(*node.arguments)

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

        LOGGER_INFERENCE.log(
            LogLevel.DEBUG_UPDATES.value,
            "- Proposal log update: {pl}\n".format(pl=proposal_log_update)
            + "- Node log update: {nl}\n".format(nl=node_log_update)
            + "- Children log updates: {cl}\n".format(cl=children_log_updates)
            + "- Is accepted: {ia}\n".format(ia=is_accepted),
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
        (
            proposed_value,
            negative_proposal_log_update,
            auxiliary_variables,
        ) = proposer.propose(node, self.world_)

        LOGGER_INFERENCE.log(
            LogLevel.DEBUG_UPDATES.value,
            "=" * 30
            + "\n"
            + "Node: {n}\n".format(n=node)
            + "- Node value: {nv}\n".format(
                # pyre-fixme
                nv=self.world_.get_node_in_world(node, False, False).value
            )
            + "- Proposed value: {pv}\n".format(pv=proposed_value),
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
        markov_blanket_func[get_wrapper(block.first_node.function)] = [block.first_node]
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
                if self.world_.is_marked_node_for_delete(node):
                    continue
                # We look up the node's current markov blanket before re-sampling
                old_node_markov_blanket = (
                    self.world_.get_markov_blanket(node) - self.observations_.keys()
                )
                proposer = self.find_best_single_site_proposer(node)
                LOGGER_PROPOSER.log(
                    LogLevel.DEBUG_PROPOSER.value,
                    "=" * 30
                    + "\n"
                    + "Proposer info for node: {n}\n".format(n=node)
                    + "- Type: {pt}\n".format(pt=str(type(proposer))),
                )
                # We use the best single site proposer to propose a new value.
                (
                    proposed_value,
                    negative_proposal_log_update,
                    auxiliary_variables,
                ) = proposer.propose(node, self.world_)
                neg_proposal_log_updates += negative_proposal_log_update

                LOGGER_INFERENCE.log(
                    LogLevel.DEBUG_UPDATES.value,
                    "Node: {n}\n".format(n=node)
                    + "- Node value: {nv}\n".format(
                        # pyre-fixme
                        nv=self.world_.get_node_in_world(node, False, False).value
                    )
                    + "- Proposed value: {pv}\n".format(pv=proposed_value),
                )

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
                new_node_markov_blanket = (
                    self.world_.get_markov_blanket(node) - self.observations_.keys()
                )
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
                    if get_wrapper(new_node.function) not in markov_blanket_func:
                        markov_blanket_func[get_wrapper(new_node.function)] = []
                    markov_blanket_func[get_wrapper(new_node.function)].append(new_node)
                markov_blanket |= new_nodes_to_be_added

        proposal_log_updates = pos_proposal_log_updates + neg_proposal_log_updates
        return nodes_log_updates, children_log_updates, proposal_log_updates

    def single_inference_run_with_sequential_block_update(self, block: Block):
        """
        Run one iteration of the inference algorithm for a given block.

        :param block: the block of random variables to be resampled sequentially
        in this inference run
        """
        LOGGER_INFERENCE.log(
            LogLevel.DEBUG_UPDATES.value,
            "=" * 30 + "\n" + "Block: {b}\n".format(b=block.first_node),
        )

        (
            nodes_log_updates,
            children_log_updates,
            proposal_log_updates,
        ) = self.block_propose_change(block)
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
        self,
        num_samples: int,
        num_adaptive_samples: int = 0,
        verbose: VerboseLevel = VerboseLevel.LOAD_BAR,
        initialize_from_prior: bool = False,
    ) -> Dict[RVIdentifier, Tensor]:
        """
        Run inference algorithms.

        :param num_samples: number of samples excluding adaptation.
        :param num_adapt_steps: number of steps to adapt/tune the proposer.
        :param verbose: Integer indicating how much output to print to stdio
        :param initialize_from_prior: boolean to initialize samples from prior
        :returns: samples for the query
        """
        self.initialize_world(initialize_from_prior)
        self.world_.set_initialize_from_prior(True)
        queries_sample = defaultdict()
        LOGGER_WORLD.log(
            LogLevel.DEBUG_GRAPH.value,
            "=" * 30 + "\n" + "Initialized graph:\n{g}\n".format(g=str(self.world_)),
        )
        # pyre-fixme
        for iteration in tqdm(
            iterable=range(num_samples + num_adaptive_samples),
            desc="Samples collected",
            disable=not verbose == VerboseLevel.LOAD_BAR,
        ):
            blocks = self.process_blocks()
            shuffle(blocks)
            for block in blocks:
                if block.type == BlockType.SINGLENODE:
                    node = block.first_node
                    if node in self.observations_ or not self.world_.contains_in_world(
                        node
                    ):
                        continue

                    proposer = self.find_best_single_site_proposer(node)

                    LOGGER_PROPOSER.log(
                        LogLevel.DEBUG_PROPOSER.value,
                        "=" * 30
                        + "\n"
                        + "Proposer info for node: {n}\n".format(n=node)
                        + "- Type: {pt}\n".format(pt=str(type(proposer))),
                    )
                    if (
                        not self.skip_single_inference_run
                        or iteration >= num_adaptive_samples
                    ):
                        is_accepted, acceptance_probability = self.single_inference_run(
                            node, proposer
                        )

                    if iteration < num_adaptive_samples:
                        if self.skip_single_inference_run:
                            is_accepted = True
                            acceptance_probability = tensor(1.0)

                        proposer.do_adaptation(
                            node,
                            self.world_,
                            acceptance_probability,
                            iteration,
                            num_adaptive_samples,
                            is_accepted,
                        )

                if (
                    block.type == BlockType.SEQUENTIAL
                    and iteration >= num_adaptive_samples
                ):
                    self.single_inference_run_with_sequential_block_update(block)

            for query in self.queries_:
                # unsqueeze the sampled value tensor, which adds an extra dimension
                # along which we'll be adding samples generated at each iteration
                if query not in queries_sample:
                    queries_sample[query] = (
                        get_wrapper(query.function)(*query.arguments)
                        .unsqueeze(0)
                        .clone()
                        .detach()
                    )
                else:
                    queries_sample[query] = torch.cat(
                        [
                            queries_sample[query],
                            get_wrapper(query.function)(*query.arguments)
                            .unsqueeze(0)
                            .clone()
                            .detach(),
                        ],
                        dim=0,
                    )
            self.world_.accept_diff()
            LOGGER_WORLD.log(
                LogLevel.DEBUG_GRAPH.value,
                "=" * 30 + "\n" + "Graph update:\n{g}\n".format(g=str(self.world_)),
            )
        return queries_sample
