# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from functools import lru_cache
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.optim as optim
from torch import Tensor, tensor
from tqdm.auto import tqdm

from ...inference import SingleSiteAncestralMetropolisHastings
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


OBS_EMBEDDING_DIM = 32  # embedding dimension for observations
NODE_EMBEDDING_DIM = 16  # embedding dimensions for node values
MB_EMBEDDING_DIM = 32  # embedding dimension for Markov blankets

LOGGER_IC = logging.getLogger("beanmachine.debug.ic")

# World + Markov Blanket + Observations -> ProposalDistribution. We need to pass
# Markov Blanket explicitly because the `World`s encountered during training
# are sampled from the generative model i.e. without conditioning on any
# `observations`
ProposerFunc = Callable[
    [World, Iterable[RVIdentifier], Dict[RVIdentifier, Tensor]], ProposalDistribution
]


class ICProposer(AbstractSingleSiteSingleStepProposer):
    _observations: Dict[RVIdentifier, Tensor] = {}

    def __init__(
        self, proposer_func: ProposerFunc, observations: Dict[RVIdentifier, Tensor]
    ):
        self._proposer_func = proposer_func
        self._observations = observations
        super().__init__()

    def get_proposal_distribution(
        self,
        node: RVIdentifier,
        node_var: Variable,
        world: World,
        auxiliary_variables: Dict,
    ) -> Tuple[ProposalDistribution, Dict]:
        if not self._observations:
            raise Exception("ICProposer must be provided observations.")
        markov_blanket = filter(
            lambda x: x not in self._observations, world.get_markov_blanket(node)
        )
        proposal_distribution = self._proposer_func(
            world, markov_blanket, self._observations
        )
        LOGGER_IC.log(
            logging.DEBUG, f"{node} proposal distribution: {proposal_distribution}"
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
    _proposers: Optional[
        Callable[[RVIdentifier], Callable[[Dict[RVIdentifier, Tensor]], ICProposer]]
    ] = None

    def find_best_single_site_proposer(
        self, node: RVIdentifier
    ) -> AbstractSingleSiteSingleStepProposer:
        proposers = self._proposers
        if proposers is not None:
            ic_proposer = proposers(node)
            if ic_proposer is not None:
                return ic_proposer(self.observations_)

        # fall back on AncestralSampler
        LOGGER_IC.warn(f"No IC artifact found for {node}, using ancestral proposer.")
        return SingleSiteAncestralProposer()

    def compile(
        self,
        observation_keys: Sequence[RVIdentifier],
        num_worlds: int = 100,
        optimizer_func=lambda parameters: optim.Adam(parameters, lr=1e-3),
    ) -> "ICInference":
        """
        Trains neural network proposers for all unobserved variables encountered
        during training.

        :param observation_keys: the nodes which are observed (must match
        those during inference)
        :param num_worlds: number of worlds drawn from the generative model
        to use for training
        :param optimizer_func: callable returning a torch.optim to optimize
        model parameters with
        """
        if len(observation_keys) == 0:
            raise Exception("Expected at least one observation RVIdentifier")
        if not all(map(lambda x: type(x) == RVIdentifier, observation_keys)):
            raise Exception("Expected every observation_key to be of type RVIdentifier")

        random_seed = torch.randint(AbstractInference._rand_int_max, (1,)).int().item()
        self.queries_ = []
        observations = {k: None for k in observation_keys}
        self.observations_ = observations
        AbstractInference.set_seed_for_chain(random_seed, 0)
        self.initialize_world(initialize_from_prior=True)

        # observation embedding is the only network that needs to be explicitly
        # constructed (the rest are lazily built)
        obs_embedding_net = self._build_observation_embedding_network(
            observation_keys, OBS_EMBEDDING_DIM
        )
        self._obs_embedding_net = obs_embedding_net

        node_embedding_nets = lru_cache(maxsize=None)(
            lambda node: self._build_node_embedding_network(
                self.world_.get_node_in_world_raise_error(node), NODE_EMBEDDING_DIM
            )
        )
        self._node_embedding_nets = node_embedding_nets

        mb_embedding_nets = lru_cache(maxsize=None)(
            lambda _: self._build_markov_blanket_embedding_network(
                MB_EMBEDDING_DIM, NODE_EMBEDDING_DIM
            )
        )
        self._mb_embedding_nets = mb_embedding_nets

        node_proposal_param_nets = lru_cache(maxsize=None)(
            lambda node: nn.Linear(
                in_features=OBS_EMBEDDING_DIM + MB_EMBEDDING_DIM,
                out_features=self._proposal_distribution_for_node(node)[0],
            )
        )
        self._node_proposal_param_nets = node_proposal_param_nets

        proposers = lru_cache(maxsize=None)(
            lambda node: lambda observations: ICProposer(
                proposer_func=self._proposer_func_for_node(node),
                observations=observations,
            )
        )
        self._proposers = proposers

        ancestral_sampler = SingleSiteAncestralMetropolisHastings()
        # pyre-fixme
        for i in tqdm(range(num_worlds)):
            # draw a world from the generative model
            ancestral_sampler.reset()
            ancestral_sampler.queries_ = observation_keys
            ancestral_sampler.observations_ = {}
            ancestral_sampler.initialize_world(initialize_from_prior=True)

            # extract parameters for all networks involved in this world's IC
            # proposers; this is necessary because networks (hence parameters)
            # can be added/removed depending on which latents are present in a
            # world
            parameters = list(obs_embedding_net.parameters())
            for node in ancestral_sampler.world_.get_all_world_vars():
                if node in observation_keys:
                    continue
                parameters.extend(
                    list(node_embedding_nets(node).parameters())
                    + list(mb_embedding_nets(node).parameters())
                    + list(node_proposal_param_nets(node).parameters())
                )

            # perform an optimization step on the parameters for networks in this world
            optimizer = optimizer_func(parameters)
            optimizer.zero_grad()
            loss = self._compute_loss(
                ancestral_sampler.world_, observation_keys, proposers
            )
            if i % int(num_worlds / 20) == 0:
                # pyre-fixme
                tqdm.write(f"Loss: {loss}")
            loss.backward()
            optimizer.step()
        ancestral_sampler.reset()
        self.reset()
        return self

    def _compute_loss(
        self,
        world: World,
        observed_rvs: Iterable[RVIdentifier],
        proposers: Callable[
            [RVIdentifier], Callable[[Dict[RVIdentifier, Tensor]], ICProposer]
        ],
    ) -> Tensor:
        observations = {
            rv: world.get_node_in_world_raise_error(rv).value for rv in observed_rvs
        }
        loss = tensor(0.0)
        for node, node_var in world.get_all_world_vars().items():
            if node in observations:
                continue
            loss -= (
                proposers(node)(observations)
                .get_proposal_distribution(node, node_var, world, {})[0]
                .proposal_distribution.log_prob(node_var.value)
                .sum()
            )
        return loss

    def _proposer_func_for_node(self, node: RVIdentifier):
        _, proposal_dist_constructor = self._proposal_distribution_for_node(node)

        def _proposer_func(
            world: World,
            markov_blanket: Iterable[RVIdentifier],
            observations: Dict[RVIdentifier, Tensor],
        ):
            obs_embedding = torch.zeros(OBS_EMBEDDING_DIM)
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

            mb_embedding = torch.zeros(MB_EMBEDDING_DIM)
            mb_nodes = list(
                map(
                    lambda mb_node: node_embedding_nets(mb_node).forward(
                        # TODO: ensure tensors exactly 1d here, OHE integers?
                        world.get_node_in_world_raise_error(mb_node)
                        .value.unsqueeze(0)
                        .float()
                    ),
                    sorted(markov_blanket, key=lambda x: str(x)),
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
                torch.cat((mb_embedding, obs_embedding))
            )

            return proposal_dist_constructor(param_vec)

        return _proposer_func

    def _proposal_distribution_for_node(
        self, node: RVIdentifier
    ) -> Tuple[int, Callable[[Tensor], dist.Distribution]]:
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
            return (2, lambda x: dist.Normal(x[0], torch.exp(x[1])))
        elif isinstance(support, dist.constraints._IntegerInterval) and isinstance(
            distribution, dist.Categorical
        ):
            num_categories = distribution.param_shape[-1]
            return (
                num_categories,
                lambda x: dist.Categorical(
                    torch.exp(x) / float(torch.exp(x).sum().item())
                ),
            )
        elif isinstance(support, dist.constraints._Boolean) and isinstance(
            distribution, dist.Bernoulli
        ):
            return (1, lambda x: dist.Bernoulli(torch.sigmoid(x)))
        else:
            raise NotImplementedError

    def _build_observation_embedding_network(
        self, observation_keys: Sequence[RVIdentifier], OBS_EMBEDDING_DIM: int
    ) -> nn.Module:
        # NOTE: assumes that observation length is the same at compile / inference
        # TODO: sequence embedding to handle variable length?

        # TODO: should just return a constant vector when observation_keys is empty
        obs_vec = torch.stack(
            list(
                map(
                    lambda node: node.function._wrapper(*node.arguments),
                    sorted(observation_keys, key=str),
                )
            ),
            dim=0,
        )
        obs_embedding_net = nn.Linear(
            in_features=obs_vec.shape[0], out_features=OBS_EMBEDDING_DIM
        )
        return obs_embedding_net

    def _build_node_embedding_network(
        self, node_var: Variable, NODE_EMBEDDING_DIM: int
    ) -> nn.Module:
        node_vec = node_var.value
        # NOTE: assumes that node does not change shape across worlds
        # TODO: better handling of 0d (e.g. tensor(1.)) vs 1d (e.g. tensor([1.]))
        in_shape = 1 if len(node_vec.shape) == 0 else node_vec.shape[0]
        node_embedding_net = nn.Linear(
            in_features=in_shape, out_features=NODE_EMBEDDING_DIM
        )
        return node_embedding_net

    def _build_markov_blanket_embedding_network(
        self, MB_EMBEDDING_DIM: int, NODE_EMBEDDING_DIM: int
    ) -> nn.Module:
        # NOTE: does NOT assume Markov Blanket shape is constant
        mb_embedding_net = nn.LSTM(
            input_size=NODE_EMBEDDING_DIM, hidden_size=MB_EMBEDDING_DIM
        )
        return mb_embedding_net
