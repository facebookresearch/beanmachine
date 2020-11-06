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
    def __init__(self, target_log_prob=None, d=1):
        super(VariationalApproximation, self).__init__()
        self.target_log_prob = target_log_prob
        self.d = d
        self.flow_stack = FlowStack(dim=self.d, n_flows=8)
    
    def arg_constraints():
        # TODO(fixme)
        return dict()

    def train(self, epochs=100, lr=1e-2):
        sample_shape = (100, self.d)
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
            loss -= self.target_log_prob(zk).sum()

            # normalize by batch size
            loss /= z0.size(0)

            loss.backward()
            optim.step()
            optim.zero_grad()

    def sample(self, sample_shape=torch.Size()):
        _, xs, _, _, _ = self.flow_stack(shape=sample_shape)
        return xs

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


class MeanFieldVariationalInference(object, metaclass=ABCMeta):
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
        num_iter: int = 100,
        lr: float = 1e-2,
    ) -> Dict[RVIdentifier, VariationalApproximation]:
        try:
            random_seed = (
                torch.randint(MeanFieldVariationalInference._rand_int_max, (1,)).int().item()
            )
            MeanFieldVariationalInference.set_seed(random_seed, 0)
            self.queries_ = queries
            self.observations_ = observations

            # TODO: handle dimension
            vi_dicts = defaultdict(VariationalApproximation)
            for iteration in tqdm(
                iterable=range(num_iter),
                desc="Training iterations",
            ):
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
                nodes = list(self.world_.get_all_world_vars().items())
                shuffle(nodes)
                # loss = tensor(0.0)
                for rvid, node_var in nodes:
                    if rvid in self.observations_:
                        continue
                    else:
                        proposer = vi_dicts[rvid]

                        def _target_log_prob(x):
                            self.world_.propose_change(
                                rvid, x, start_new_diff=True
                            )
                            log_prob = node_var.distribution.log_prob(x)
                            for child in node_var.children:
                                child_var = self.world_.get_node_in_world_raise_error(child)
                                log_prob += child_var.log_prob
                            self.world_.reject_diff()
                            return log_prob
                        proposer.target_log_prob = _target_log_prob
                        proposer.train(epochs=1, lr=lr)

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