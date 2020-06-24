# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.optim as optim
from torch import Tensor, tensor
from tqdm.auto import tqdm

from ...inference.abstract_infer import AbstractInference
from ...inference.abstract_mh_infer import AbstractMHInference
from ...inference.proposer.abstract_single_site_single_step_proposer import (
    AbstractSingleSiteSingleStepProposer,
)
from ...inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from ...model.utils import RVIdentifier
from ...world import ProposalDistribution, Variable, World


LOGGER_IC = logging.getLogger("beanmachine.debug.ic")

# World + Markov Blanket + Observations -> ProposalDistribution. We need to pass
# Markov Blanket explicitly because the `World`s encountered during training
# are sampled from the generative model i.e. without conditioning on any
# `observations`
ProposerFunc = Callable[
    [World, Iterable[RVIdentifier], Dict[RVIdentifier, Tensor]], ProposalDistribution
]


class ICProposer(AbstractSingleSiteSingleStepProposer):
    """
    Inference Compilation Proposer.
    """

    def __init__(self, proposer_func: ProposerFunc):
        self._proposer_func = proposer_func
        super().__init__()

    def get_proposal_distribution(
        self,
        node: RVIdentifier,
        node_var: Variable,
        world: World,
        auxiliary_variables: Dict,
    ) -> Tuple[ProposalDistribution, Dict]:
        observations = world.observations_
        markov_blanket = filter(
            lambda x: x not in observations, world.get_markov_blanket(node)
        )
        proposal_distribution = self._proposer_func(world, markov_blanket, observations)
        LOGGER_IC.log(
            logging.DEBUG,
            f"{node}={node_var.value} proposing with {proposal_distribution}",
        )
        return (
            ProposalDistribution(
                proposal_distribution=proposal_distribution,
                requires_transform=False,
                requires_reshape=False,
                arguments={},
            ),
            {},
        )


class ICInference(AbstractMHInference):
    """
    Inference compilation
    """

    _obs_embedding_net: Optional[nn.Module] = None
    _node_embedding_nets: Optional[Callable[[RVIdentifier], nn.Module]] = None
    _mb_embedding_nets: Optional[Callable[[RVIdentifier], nn.Module]] = None
    _node_proposal_param_nets: Optional[Callable[[RVIdentifier], nn.Module]] = None
    _proposers: Optional[Callable[[RVIdentifier], ICProposer]] = None
    _node_ids: List[RVIdentifier] = []
    _NODE_ID_EMBEDDING_DIM: int = 32  # embedding dimension for RVIdentifier
    _NODE_EMBEDDING_DIM: int = 4  # embedding dimension for node values
    _OBS_EMBEDDING_DIM = 4  # embedding dimension for observations
    _MB_EMBEDDING_DIM = 32  # embedding dimension for Markov blankets

    def find_best_single_site_proposer(
        self, node: RVIdentifier
    ) -> AbstractSingleSiteSingleStepProposer:
        proposers = self._proposers
        if proposers is not None:
            ic_proposer = proposers(node)
            if ic_proposer is not None:
                return ic_proposer

        # fall back on AncestralSampler
        LOGGER_IC.warn(f"No IC artifact found for {node}, using ancestral proposer.")
        return SingleSiteAncestralProposer()

    def compile(
        self,
        observation_keys: Sequence[RVIdentifier],
        num_worlds: int = 100,
        batch_size: int = 16,
        optimizer_func=lambda parameters: optim.Adam(parameters, lr=1e-3),
        max_num_rvs: int = 32,
    ) -> "ICInference":
        """
        Trains neural network proposers for all unobserved variables encountered
        during training.

        :param observation_keys: the nodes which are observed (must match
        those during inference)
        :param num_worlds: number of worlds drawn from the generative model
        and used for training
        :param batch_size: number of worlds used in each optimization step
        :param optimizer_func: callable returning a torch.optim to optimize
        model parameters with
        :param max_num_rvs: RVIdentifier OHE embedding dimension, must upper
        bound the number of unique RVs in any world
        """
        if len(observation_keys) == 0:
            raise Exception("Expected at least one observation RVIdentifier")
        if not all(map(lambda x: type(x) == RVIdentifier, observation_keys)):
            raise Exception("Expected every observation_key to be of type RVIdentifier")

        self._NODE_ID_EMBEDDING_DIM = max_num_rvs

        random_seed = torch.randint(AbstractInference._rand_int_max, (1,)).int().item()
        AbstractInference.set_seed_for_chain(random_seed, 0)

        # used for assigning unique sequential IDs to OHE embed RVIdentifiers
        # as they are encountered
        self._node_ids = []

        # initialize once so observation embedding network can access RVIdentifiers
        self.reset()
        self.queries_ = {}
        self.observations_ = {obs_rv: None for obs_rv in observation_keys}
        self.initialize_world(initialize_from_prior=True)

        # observation embedding is the only network that needs to be explicitly
        # constructed (the rest are lazily built)
        obs_embedding_net = self._build_observation_embedding_network(observation_keys)
        self._obs_embedding_net = obs_embedding_net

        node_embedding_nets = lru_cache(maxsize=None)(
            self._build_node_embedding_network
        )
        self._node_embedding_nets = node_embedding_nets

        mb_embedding_nets = lru_cache(maxsize=None)(
            lambda _: nn.LSTM(
                input_size=self._NODE_EMBEDDING_DIM + self._NODE_ID_EMBEDDING_DIM,
                num_layers=3,
                hidden_size=self._MB_EMBEDDING_DIM,
            )
        )
        self._mb_embedding_nets = mb_embedding_nets

        node_proposal_param_nets = lru_cache(maxsize=None)(
            lambda node: nn.Sequential(
                nn.Linear(
                    in_features=self._NODE_ID_EMBEDDING_DIM
                    + self._NODE_EMBEDDING_DIM
                    + self._MB_EMBEDDING_DIM
                    + self._OBS_EMBEDDING_DIM,
                    out_features=self._proposal_distribution_for_node(node)[0],
                ),
                nn.Tanh(),
                nn.Linear(
                    in_features=self._proposal_distribution_for_node(node)[0],
                    out_features=self._proposal_distribution_for_node(node)[0],
                ),
            )
        )
        self._node_proposal_param_nets = node_proposal_param_nets

        proposers = lru_cache(maxsize=None)(
            lambda node: ICProposer(proposer_func=self._proposer_func_for_node(node))
        )
        self._proposers = proposers

        optimizer = optimizer_func(obs_embedding_net.parameters())
        rvs_in_optimizer = set()

        num_batches = int(math.ceil(num_worlds / batch_size))
        # pyre-fixme
        for i in tqdm(range(num_batches)):
            optimizer.zero_grad()
            loss = torch.zeros(1)

            for _ in range(batch_size):
                # draw a World from the generative model
                self.reset()
                self.queries_ = observation_keys
                self.observations_ = {}
                self.initialize_world(initialize_from_prior=True)

                # add parameters for node-specific NNs associated with any new nodes
                # encountered in this World
                for node in self.world_.get_all_world_vars():
                    if (node not in observation_keys) and (
                        node not in rvs_in_optimizer
                    ):
                        LOGGER_IC.debug(
                            f"Adding {node} neural network parameters to optimizer"
                        )
                        rvs_in_optimizer.add(node)
                        optimizer.add_param_group(
                            {
                                "params": (
                                    list(node_embedding_nets(node).parameters())
                                    + list(mb_embedding_nets(node).parameters())
                                    + list(node_proposal_param_nets(node).parameters())
                                )
                            }
                        )
                # we set the world's observations here because loss is computed
                # only for nodes not in world._observations_
                observations = {
                    rv: self.world_.get_node_in_world_raise_error(rv).value
                    for rv in observation_keys
                }
                self.world_.set_observations(observations)
                loss += self._compute_loss(self.world_, proposers)
            loss.backward()
            optimizer.step()

            print_every = int(num_batches / 20)
            if (print_every == 0) or (i % print_every == 0):
                # pyre-fixme
                tqdm.write(f"Loss: {loss}")
        self.reset()
        return self

    def _compute_loss(
        self, world: World, proposers: Callable[[RVIdentifier], ICProposer]
    ) -> Tensor:
        loss = tensor(0.0)
        for node, node_var in world.get_all_world_vars().items():
            if node in world.observations_:
                continue
            loss -= (
                proposers(node)
                .get_proposal_distribution(node, node_var, world, {})[0]
                .proposal_distribution.log_prob(node_var.value)
                .sum()
            )
        return loss

    def _build_observation_embedding_network(
        self, observation_keys: Sequence[RVIdentifier]
    ) -> nn.Module:
        obs_vec = torch.stack(
            list(
                map(
                    lambda node: node.function._wrapper(*node.arguments),
                    sorted(observation_keys, key=str),
                )
            ),
            dim=0,
        )
        return nn.Linear(
            in_features=obs_vec.shape[0], out_features=self._OBS_EMBEDDING_DIM
        )

    def _build_node_embedding_network(self, node: RVIdentifier) -> nn.Module:
        node_var = self.world_.get_node_in_world_raise_error(node)
        node_vec = node_var.value
        # NOTE: assumes that node does not change shape across worlds
        # TODO: better handling of 0d (e.g. tensor(1.)) vs 1d (e.g. tensor([1.]))
        in_shape = 1 if len(node_vec.shape) == 0 else node_vec.shape[0]
        node_embedding_net = nn.Sequential(
            nn.Linear(in_features=in_shape, out_features=self._NODE_EMBEDDING_DIM)
        )

        # explicitly encode node id, c.f. "address" in trace-based IC
        try:
            node_id = self._node_ids.index(node)
        except ValueError:
            node_id = len(self._node_ids)
            self._node_ids.append(node)
        node_id_ohe = torch.zeros(self._NODE_ID_EMBEDDING_DIM)
        node_id_ohe[node_id] = 1.0

        class NodeEmbedding(nn.Module):
            """
            Node embedding network which concatenates one-hot encoding of
            node ID with node value embedding.
            """

            def __init__(self, id_ohe, embedding_net):
                super().__init__()
                self.id_ohe = id_ohe
                self.embedding_net = embedding_net

            def forward(self, x):
                return torch.cat((self.id_ohe, self.embedding_net.forward(x.float())))

        return NodeEmbedding(node_id_ohe, node_embedding_net)

    def _proposer_func_for_node(self, node: RVIdentifier):
        _, proposal_dist_constructor = self._proposal_distribution_for_node(node)

        def _proposer_func(
            world: World,
            markov_blanket: Iterable[RVIdentifier],
            observations: Dict[RVIdentifier, Tensor],
        ) -> dist.Distribution:
            obs_embedding = torch.zeros(self._OBS_EMBEDDING_DIM)
            obs_nodes = list(
                map(
                    lambda x: x[1],
                    sorted(observations.items(), key=lambda x: str(x[0])),
                )
            )
            if len(obs_nodes):
                obs_embedding_net = self._obs_embedding_net
                if obs_embedding_net is None:
                    raise Exception("No observation embedding network found!")

                obs_vec = torch.stack(obs_nodes, dim=0)
                # pyre-fixme
                obs_embedding = obs_embedding_net.forward(obs_vec)

            node_embedding_nets = self._node_embedding_nets
            if node_embedding_nets is None:
                raise Exception("No node embedding networks found!")

            node_embedding = node_embedding_nets(node).forward(
                # TODO: ensure tensors exactly 1d here, OHE integers?
                world.get_node_in_world_raise_error(node).value.unsqueeze(0)
            )

            mb_embedding = torch.zeros(self._MB_EMBEDDING_DIM)
            mb_nodes = list(
                map(
                    lambda mb_node: node_embedding_nets(mb_node).forward(
                        # TODO: ensure tensors exactly 1d here, OHE integers?
                        world.get_node_in_world_raise_error(mb_node)
                        .value.unsqueeze(0)
                        .float()
                    ),
                    sorted(markov_blanket, key=str),
                )
            )
            if len(mb_nodes):
                mb_embedding_nets = self._mb_embedding_nets
                if mb_embedding_nets is None:
                    raise Exception("No Markov blanket embedding networks found!")

                # NOTE: currently adds batch axis (at index 1) here, may need
                # to change when we batch training (see
                # torch.nn.utils.rnn.PackedSequence)
                mb_vec = torch.stack(mb_nodes, dim=0).unsqueeze(1)
                # TODO: try pooling rather than just slicing out last hidden
                mb_embedding = (
                    mb_embedding_nets(node).forward(mb_vec)[0][-1, :, :].squeeze()
                )
            node_proposal_param_nets = self._node_proposal_param_nets
            if node_proposal_param_nets is None:
                raise Exception("No node proposal parameter networks found!")
            param_vec = node_proposal_param_nets(node).forward(
                torch.cat((node_embedding, mb_embedding, obs_embedding))
            )
            return proposal_dist_constructor(param_vec)

        return _proposer_func

    def _proposal_distribution_for_node(
        self, node: RVIdentifier
    ) -> Tuple[int, Callable[[Tensor], dist.Distribution]]:
        """
        :param node: random variable to build parameterized proposal distribution for
        :returns: num_parameters, constructor function taking a (num_parameters,) shaped
        Tensor and initializing the appropriate proposal distribution for `node`
        """
        node_var = self.world_.get_node_in_world_raise_error(node)
        distribution = node_var.distribution
        # pyre-fixme
        support = distribution.support
        # NOTE: only univariates supported
        if (
            isinstance(support, dist.constraints._Real)
            or isinstance(support, dist.constraints._Simplex)
            or isinstance(support, dist.constraints._GreaterThan)
        ):
            # TODO: use a GMM
            return (2, lambda x: dist.Normal(loc=x[0], scale=torch.exp(x[1])))
        elif isinstance(support, dist.constraints._IntegerInterval) and isinstance(
            distribution, dist.Categorical
        ):
            num_categories = distribution.param_shape[-1]
            return (num_categories, lambda x: dist.Categorical(logits=x))
        elif isinstance(support, dist.constraints._Boolean) and isinstance(
            distribution, dist.Bernoulli
        ):
            return (1, lambda x: dist.Bernoulli(logits=x))
        else:
            raise NotImplementedError
