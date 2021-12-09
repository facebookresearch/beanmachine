# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from functools import lru_cache
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.optim as optim
from torch import Tensor, tensor
from tqdm.auto import tqdm

from ...legacy.inference.abstract_infer import (
    AbstractInference,
    AbstractMCInference,
)
from ...legacy.inference.abstract_mh_infer import AbstractMHInference
from ...legacy.inference.proposer.abstract_single_site_single_step_proposer import (
    AbstractSingleSiteSingleStepProposer,
)
from ...legacy.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from ...legacy.world import ProposalDistribution, Variable, World
from ...model.rv_identifier import RVIdentifier
from ...world.utils import is_constraint_eq
from . import utils


LOGGER_IC = logging.getLogger("beanmachine.debug.ic")

# World + Markov Blanket + Observations -> ProposalDistribution. We need to pass
# Markov Blanket explicitly because the `World`s encountered during training
# are sampled from the generative model i.e. without conditioning on any
# `observations`
ProposerFunc = Callable[
    [World, Iterable[RVIdentifier], Dict[RVIdentifier, Tensor]], dist.Distribution
]


class ICProposer(AbstractSingleSiteSingleStepProposer):
    """
    Inference Compilation Proposer.
    """

    def __init__(self, proposer_func: ProposerFunc, optimizer: optim.Optimizer):
        self._proposer_func: ProposerFunc = proposer_func
        self._optimizer: optim.Optimizer = optimizer
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

    def do_adaptation(
        self,
        node: RVIdentifier,
        world: World,
        acceptance_probability: Tensor,
        iteration_number: int,
        num_adaptive_samples: int,
        is_accepted: bool,
    ) -> None:
        """
        Adapts inference compilation (IC) proposers by hill climbing to improve
        proposal probability of the provided `node`'s value. As `world` consists
        of accepted MH samples, we can view IC adaptation as hill climbing an
        inclusive KL-divergence computed empirically over samples (x,y) where the
        observations y are consistent with test-time distributions (e.g.
        covariate shift).

        :param node: the node in `world` to perform proposer adaptation for
        :param world: the new world if `is_accepted`, or the previous world
        otherwise.
        :param acceptance_probability: the acceptance probability of the previous move.
        :param iteration_number: the current iteration of inference
        :param num_adaptive_samples: the number of inference iterations for adaptation.
        :param is_accepted: bool representing whether the new value was accepted.
        :returns: nothing.
        """
        node_var = world.get_node_in_world_raise_error(node)
        markov_blanket = filter(
            lambda x: x not in world.observations_, world.get_markov_blanket(node)
        )
        proposal_distribution = self._proposer_func(
            world, markov_blanket, world.observations_
        )

        optimizer = self._optimizer
        loss = -(proposal_distribution.log_prob(node_var.value))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class ICInference(AbstractMHInference):
    """
    Inference compilation
    """

    _obs_embedding_net: Optional[nn.Module] = None
    _node_embedding_nets: Optional[Callable[[RVIdentifier], nn.Module]] = None
    _mb_embedding_nets: Optional[Callable[[RVIdentifier], nn.Module]] = None
    _node_proposal_param_nets: Optional[Callable[[RVIdentifier], nn.Module]] = None
    _optimizer: Optional[optim.Optimizer] = None
    _proposers: Optional[Callable[[RVIdentifier], ICProposer]] = None
    _GMM_NUM_COMPONENTS: int = 1  # number of components in GMM density estimators
    _NODE_ID_EMBEDDING_DIM: int = 0  # embedding dimension for RVIdentifier
    _NODE_EMBEDDING_DIM: int = 4  # embedding dimension for node values
    _OBS_EMBEDDING_DIM: int = 4  # embedding dimension for observations
    _MB_EMBEDDING_DIM: int = 8  # embedding dimension for Markov blankets
    _MB_NUM_LAYERS = 3  # num LSTM layers for Markov blankets
    _NODE_PROPOSAL_NUM_LAYERS = 1  # num layers for node proposal parameter nets
    _ENTROPY_REGULARIZATION_COEFFICIENT: float = 0.0

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

    def compile(  # noqa: C901
        self,
        observation_keys: Sequence[RVIdentifier],
        num_worlds: int = 100,
        batch_size: int = 16,
        optimizer_func=lambda parameters: optim.Adam(parameters),
        node_id_embedding_dim: Optional[int] = None,
        node_embedding_dim: Optional[int] = None,
        obs_embedding_dim: Optional[int] = None,
        mb_embedding_dim: Optional[int] = None,
        mb_num_layers: Optional[int] = None,
        node_proposal_num_layers: Optional[int] = None,
        entropy_regularization_coefficient: Optional[float] = None,
        gmm_num_components: Optional[int] = None,
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
        :param node_id_embedding_dim: RVIdentifier ID embedding dimension
        :param node_embedding_dim: RVIdentifier embedding dimension
        :param obs_embedding_dim: observations embedding dimension
        :param mb_embedding_dim: Markov blanket embedding dimension
        :param mb_num_layers: number of layers in Markov blanket embedding RNN
        :param node_proposal_num_layers: number of layers in proposal parameter FFW NN
        :param entropy_regularization_coefficient: coefficient for entropy regularization
        term in training loss function
        :param gmm_num_components: number of components in GMM density estimators
        """
        if len(observation_keys) == 0:
            raise Exception("Expected at least one observation RVIdentifier")
        if not all(map(lambda x: type(x) == RVIdentifier, observation_keys)):
            raise Exception("Expected every observation_key to be of type RVIdentifier")

        if node_id_embedding_dim:
            self._NODE_ID_EMBEDDING_DIM = node_id_embedding_dim
        if node_embedding_dim:
            self._NODE_EMBEDDING_DIM = node_embedding_dim
        if obs_embedding_dim:
            self._OBS_EMBEDDING_DIM = obs_embedding_dim
        if mb_embedding_dim:
            self._MB_EMBEDDING_DIM = mb_embedding_dim
        if mb_num_layers:
            self._MB_NUM_LAYERS = mb_num_layers
        if node_proposal_num_layers:
            self._NODE_PROPOSAL_NUM_LAYERS = node_proposal_num_layers
        if entropy_regularization_coefficient:
            self._ENTROPY_REGULARIZATION_COEFFICIENT = (
                entropy_regularization_coefficient
            )
        if gmm_num_components:
            self._GMM_NUM_COMPONENTS = gmm_num_components

        random_seed = torch.randint(AbstractInference._rand_int_max, (1,)).int().item()
        AbstractMCInference.set_seed_for_chain(random_seed, 0)

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
                num_layers=self._MB_NUM_LAYERS,
                hidden_size=self._MB_EMBEDDING_DIM,
            )
        )
        self._mb_embedding_nets = mb_embedding_nets

        node_proposal_param_nets = lru_cache(maxsize=None)(
            self._build_node_proposal_param_network
        )
        self._node_proposal_param_nets = node_proposal_param_nets

        optimizer = optimizer_func(obs_embedding_net.parameters())
        if not optimizer:
            raise Exception("optimizer_func did not return a valid optimizer!")
        self._optimizer = optimizer
        proposers = lru_cache(maxsize=None)(
            lambda node: ICProposer(
                proposer_func=self._proposer_func_for_node(node), optimizer=optimizer
            )
        )
        self._proposers = proposers

        rvs_in_optimizer = set()

        num_batches = int(math.ceil(num_worlds / batch_size))
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
                                    + list(node_proposal_param_nets(node).parameters())
                                    + (
                                        list(mb_embedding_nets(node).parameters())
                                        if self._MB_EMBEDDING_DIM > 0
                                        else []
                                    )
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

            # only back-propagate if backward() is available to handle corner case
            # where no `queries` are present in this compilation
            if loss.requires_grad:
                loss.backward()
                optimizer.step()

            print_every = int(num_batches / 20)
            if (print_every == 0) or (i % print_every == 0):
                tqdm.write(f"Loss: {loss}", end="")
        self.reset()
        return self

    def _compute_loss(
        self, world: World, proposers: Callable[[RVIdentifier], ICProposer]
    ) -> Tensor:
        loss = tensor(0.0)
        for node, node_var in world.get_all_world_vars().items():
            if node in world.observations_:
                continue
            proposal_distribution = (
                proposers(node)
                .get_proposal_distribution(node, node_var, world, {})[0]
                .proposal_distribution
            )
            loss -= proposal_distribution.log_prob(node_var.value).sum()
            if self._ENTROPY_REGULARIZATION_COEFFICIENT > 0.0:
                loss += (
                    self._ENTROPY_REGULARIZATION_COEFFICIENT
                    * proposal_distribution.entropy()
                )

        return loss

    def _build_observation_embedding_network(
        self, observation_keys: Sequence[RVIdentifier]
    ) -> nn.Module:
        obs_vec = torch.stack(
            list(
                map(
                    lambda node: self.world_.call(node),
                    sorted(observation_keys, key=str),
                )
            ),
            dim=0,
        ).flatten()
        return nn.Linear(
            in_features=obs_vec.shape[0], out_features=self._OBS_EMBEDDING_DIM
        )

    def _build_node_embedding_network(self, node: RVIdentifier) -> nn.Module:
        node_var = self.world_.get_node_in_world_raise_error(node)
        node_vec = utils.ensure_1d(node_var.value)
        # NOTE: assumes that node does not change shape across worlds
        node_embedding_net = nn.Sequential(
            nn.Linear(
                in_features=node_vec.shape[0], out_features=self._NODE_EMBEDDING_DIM
            )
        )
        node_id_embedding = torch.randn(self._NODE_ID_EMBEDDING_DIM)
        node_id_embedding /= node_id_embedding.norm(p=2)

        class NodeEmbedding(nn.Module):
            """
            Node embedding network which concatenates one-hot encoding of
            node ID with node value embedding.
            """

            def __init__(self, node_id_embedding, embedding_net):
                super().__init__()
                self.node_id_embedding = node_id_embedding
                self.embedding_net = embedding_net

            def forward(self, x):
                return torch.cat(
                    (self.node_id_embedding, self.embedding_net.forward(x.float()))
                )

        return NodeEmbedding(node_id_embedding, node_embedding_net)

    def _build_node_proposal_param_network(self, node: RVIdentifier) -> nn.Module:
        in_features = self._MB_EMBEDDING_DIM + self._OBS_EMBEDDING_DIM
        layers = []
        for _ in range(self._NODE_PROPOSAL_NUM_LAYERS):
            # TODO: bottlenecking?
            layers.extend(
                [nn.Linear(in_features=in_features, out_features=in_features), nn.ELU()]
            )
        layers.append(
            nn.Linear(
                in_features=in_features,
                out_features=self._proposal_distribution_for_node(node)[0],
            )
        )
        return nn.Sequential(*layers)

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

                obs_vec = torch.stack(obs_nodes, dim=0).flatten()
                # pyre-fixme
                obs_embedding = obs_embedding_net.forward(obs_vec)

            node_embedding_nets = self._node_embedding_nets
            if node_embedding_nets is None:
                raise Exception("No node embedding networks found!")

            mb_embedding = torch.zeros(self._MB_EMBEDDING_DIM)
            mb_nodes = list(
                map(
                    # pyre-fixme[29]: `Union[Tensor, nn.Module]` is not a function.
                    lambda mb_node: node_embedding_nets(mb_node).forward(
                        utils.ensure_1d(
                            world.get_node_in_world_raise_error(mb_node).value
                        )
                    ),
                    sorted(markov_blanket, key=str),
                )
            )
            if len(mb_nodes) and self._MB_EMBEDDING_DIM > 0:
                mb_embedding_nets = self._mb_embedding_nets
                if mb_embedding_nets is None:
                    raise Exception("No Markov blanket embedding networks found!")

                # NOTE: currently adds batch axis (at index 1) here, may need
                # to change when we batch training (see
                # torch.nn.utils.rnn.PackedSequence)
                mb_vec = torch.stack(mb_nodes, dim=0).unsqueeze(1)
                # TODO: try pooling rather than just slicing out last hidden
                mb_embedding = utils.ensure_1d(
                    # pyre-fixme[29]: `Union[Tensor, nn.Module]` is not a function.
                    mb_embedding_nets(node)
                    .forward(mb_vec)[0][-1, :, :]
                    .squeeze()
                )
            node_proposal_param_nets = self._node_proposal_param_nets
            if node_proposal_param_nets is None:
                raise Exception("No node proposal parameter networks found!")
            # pyre-fixme[29]: `Union[Tensor, nn.Module]` is not a function.
            param_vec = node_proposal_param_nets(node).forward(
                torch.cat((mb_embedding, obs_embedding))
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
        sample_val = distribution.sample()
        # pyre-fixme
        support = distribution.support

        ndim = sample_val.dim()
        if ndim > 1:
            raise NotImplementedError(
                f"IC currently only supports 0D (scalar) and 1D (vector) values. "
                f"Encountered node={node} with dim={ndim}"
            )

        if any(
            is_constraint_eq(
                support,
                (
                    dist.constraints.real,
                    dist.constraints.real_vector,
                    dist.constraints.simplex,
                    dist.constraints.greater_than,
                ),
            )
        ):
            k = self._GMM_NUM_COMPONENTS
            if ndim == 0:

                def _func(x):
                    mix = dist.Categorical(logits=x[:k])
                    comp = dist.Independent(
                        dist.Normal(
                            loc=x[k : 2 * k], scale=torch.exp(x[2 * k : 3 * k])
                        ),
                        reinterpreted_batch_ndims=0,
                    )
                    return dist.MixtureSameFamily(mix, comp)

                return (3 * k, _func)
            else:
                # TODO: non-diagonal covariance
                d = sample_val.shape[0]

                def _func(x):
                    mix = dist.Categorical(logits=x[:k])
                    comp = dist.Independent(
                        dist.Normal(
                            loc=x[k : k + k * d].reshape(k, d),
                            scale=torch.exp(x[k + k * d : k + 2 * k * d]).reshape(k, d),
                        ),
                        reinterpreted_batch_ndims=1,
                    )
                    return dist.MixtureSameFamily(mix, comp)

                return (k + 2 * k * d, _func)
        elif is_constraint_eq(
            support, dist.constraints.integer_interval
        ) and isinstance(distribution, dist.Categorical):
            num_categories = distribution.param_shape[-1]
            return (num_categories, lambda x: dist.Categorical(logits=x))
        elif is_constraint_eq(support, dist.constraints.boolean) and isinstance(
            distribution, dist.Bernoulli
        ):
            return (1, lambda x: dist.Bernoulli(logits=x.item()))
        else:
            raise NotImplementedError
