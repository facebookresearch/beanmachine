import random
from abc import ABCMeta
from collections import defaultdict
from random import shuffle
from typing import ClassVar, Dict, List

import torch
import torch.optim
import torch.distributions as dist
from torch import Tensor, tensor
from tqdm.auto import tqdm

from beanmachine.ppl.world import World
from beanmachine.ppl.model.statistical_model import StatisticalModel
from beanmachine.ppl.model.utils import Mode, RVIdentifier, get_wrapper

from .IAF import FlowStack


class VariationalApproximation(dist.distribution.Distribution):
    def __init__(self, d=1, target=None):
        super(VariationalApproximation, self).__init__()
        self.target = target
        self.d = d
        self.flow_stack = FlowStack(dim=2, n_flows=8)

    def train(self, epochs=100, lr=1e-2):
        sample_shape = (100, 2)
        optim = torch.optim.Adam(self.flow_stack.parameters(), lr=lr)

        for i in range(epochs):
            z0, zk, mu, log_var, ldj = self.flow_stack(shape=sample_shape)

            # negative ELBO loss

            # entropy H(Q)
            loss = (
                dist.Normal(mu, torch.exp(0.5 * log_var)).log_prob(z0).sum()
            )  # Q(z_0)
            loss -= ldj.sum()  # transport to Q(z_k) via - sum log det jac

            # negative cross-entropy -H(Q,P)
            loss -= self.target.log_prob(zk).sum()

            # normalize by batch size
            loss /= z0.size(0)

            loss.backward()
            optim.step()
            optim.zero_grad()
            if i % 100 == 0:
                print(loss.item())

    def sample(self, sample_shape=torch.Size()):
        _, xs, _, _, _ = self.flow_stack(shape=sample_shape)
        return xs.detach()

    def parameters(self):
        return self.flow_stack.parameters()

    def log_prob(self, value):
        # if z' = f(z), Q(z') = Q(z) |det df/dz|^{-1}
        z = value
        ldj = 0.0
        for flow in reversed(self.flow_stack.flow):
            h = torch.zeros(flow.context_size)
            s_t = flow.s_t(z, h) + 1.5
            sigma_t = torch.sigmoid(s_t)
            m_t = flow.m_t(z, h)
            ldj += flow.log_det_jac(sigma_t)
            z_prev = (z - (1 - sigma_t) * m_t) / sigma_t
            z = z_prev
        return (
            dist.Independent(
                dist.Normal(self.flow_stack.mu, torch.exp(self.flow_stack.log_var / 2)),
                1,
            ).log_prob(z)
            - ldj
        )


class VariationalInference(object, metaclass=ABCMeta):
    world_: World
    _rand_int_max: ClassVar[int] = 2 ** 62

    def __init__(self):
        self.world_ = World()
        self.initial_world_ = self.world_
        StatisticalModel.reset()
        self.queries_ = []
        self.observations_ = {}

    @staticmethod
    def set_seed(random_seed: int, chain: int):
        torch.manual_seed(random_seed + chain * 31)
        random.seed(random_seed + chain * 31)

    def infer(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, Tensor],
    ) -> Dict[RVIdentifier, VariationalApproximation]:
        try:
            random_seed = (
                torch.randint(VariationalInference._rand_int_max, (1,)).int().item()
            )
            VariationalInference.set_seed(random_seed, 0)
            self.queries_ = queries
            self.observations_ = observations

            # TODO: handle dimension
            vi_dicts = defaultdict(VariationalApproximation)
            for iteration in tqdm(
                iterable=range(100),
                desc="Training iterations",
            ):
                # neg ELBO loss -E_Q[log P - log Q]
                # 1) sample jointly from Q to propose a world

                # initialize world
                self.initialize_infer()
                self.world_.set_observations(self.observations_)
                StatisticalModel.set_mode(Mode.INFERENCE)
                for node in self.observations_:
                    get_wrapper(node.function)(*node.arguments)
                for node in self.queries_:
                    get_wrapper(node.function)(*node.arguments)
                self.world_.accept_diff()

                # propose each node using Variational approx, accumulating logQ/logP
                for first_node in self.queries_:
                    nodes = list(
                        map(
                            lambda x: get_wrapper(x.function),
                            self.world_.get_all_world_vars().keys(),
                        )
                    )
                    shuffle(nodes)
                    markov_blanket = set({first_node})
                    markov_blanket_func = {}
                    markov_blanket_func[get_wrapper(first_node.function)] = [first_node]
                    logQ, logP = (
                        tensor(0.0),
                        tensor(0.0),
                    )
                    for node_func in nodes:
                        if node_func not in markov_blanket_func:
                            continue
                        for node in markov_blanket_func[node_func].copy():
                            if self.world_.is_marked_node_for_delete(node):
                                continue
                            old_node_markov_blanket = (
                                self.world_.get_markov_blanket(node)
                                - self.observations_.keys()
                            )
                            proposer = vi_dicts[node]
                            proposer.target = self.world_.get_node_in_world_raise_error(
                                node
                            ).distribution
                            proposer.train()
                            proposed_value = proposer.sample(sample_shape=(1, 1))
                            logQ += proposer.log_prob(proposed_value).sum()

                            # We update the world (through a new diff in the diff stack).
                            (
                                children_log_update,
                                _,
                                node_log_update,
                                _,
                            ) = self.world_.propose_change(
                                node, proposed_value, start_new_diff=True
                            )
                            logP += children_log_update
                            logP += node_log_update
                            # We look up the updated markov blanket of the re-sampled node.
                            new_node_markov_blanket = (
                                self.world_.get_markov_blanket(node)
                                - self.observations_.keys()
                            )
                            all_node_markov_blanket = (
                                old_node_markov_blanket | new_node_markov_blanket
                            )
                            # new_nodes_to_be_added is all the new nodes to be added to
                            # entire markov blanket.
                            new_nodes_to_be_added = (
                                all_node_markov_blanket - markov_blanket
                            )
                            for new_node in new_nodes_to_be_added:
                                if new_node is None:
                                    continue
                                # We create a dictionary from node family to the node itself
                                # as the match with block happens at the family level and
                                # this makes the lookup much faster.
                                if (
                                    get_wrapper(new_node.function)
                                    not in markov_blanket_func
                                ):
                                    markov_blanket_func[
                                        get_wrapper(new_node.function)
                                    ] = []
                                markov_blanket_func[
                                    get_wrapper(new_node.function)
                                ].append(new_node)
                            markov_blanket |= new_nodes_to_be_added

        except BaseException as x:
            raise x
        finally:
            self.reset()
        return vi_dicts

    def initialize_infer(self):
        """
        Initialize inference
        """
        self.initial_world_ = self.world_.copy()
        StatisticalModel.set_world(self.world_)

    def reset(self):
        """
        Resets world, mode and observation
        """
        self.world_ = self.initial_world_
        StatisticalModel.reset()
        self.queries_ = []
        self.observations_ = {}