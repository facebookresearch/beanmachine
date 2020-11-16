import random
from abc import ABCMeta
from collections import defaultdict
from random import shuffle
from typing import ClassVar, Dict, List, Optional

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
    def __init__(self, target_log_prob=None, num_flows=8, lr=1e-2, base_dist=dist.Normal, base_args={}):
        #assert len(base_dist.event_shape) <= 1, "VariationalApproximation currently only supports 0D and 1D tensors"
        super(VariationalApproximation, self).__init__()
        self.target_log_prob = target_log_prob
        self.flow_stack = FlowStack(num_flows=num_flows, base_dist=base_dist, base_args=base_args)
        self.optim = torch.optim.Adam(
            list(self.flow_stack.parameters()) + list(self.flow_stack.base_args.values()), 
            lr=lr)
        self.has_rsample = True
    
    def arg_constraints():
        # TODO(fixme)
        return dict()

    def train(self, epochs=100):
        optim = self.optim
        for i in range(epochs):
            z0, zk, mu, log_var, ldj = self.flow_stack(shape=(100,))
            n, d = z0.shape
            # std = torch.exp(0.5 * log_var)

            # negative ELBO loss

            # entropy H(Q)
            # base_args = {mu, sigma} for normal, {nu, mu, sigma} for StudentT
            loss = self.flow_stack.base_dist(**self.flow_stack.base_args).log_prob(z0).sum()

            #loss = self.flow_stack.base_dist.log_prob((z0 - mu) / std).sum()  # Q((z_0 - mu)/sigma)
            #loss -= n * log_var.sum() / 2.0  # jac from standardizing z0, TODO: only valid for normal

            loss -= ldj.sum()  # change of variable zk -> z0

            # negative cross-entropy -H(Q,P)
            loss -= self.target_log_prob(zk).sum()

            # normalize by batch size
            loss /= z0.size(0)
            if not torch.isnan(loss):
                loss.backward(retain_graph=True)
                optim.step()
                optim.zero_grad()

    def rsample(self, sample_shape=torch.Size()):
        _, xs, _, _, _ = self.flow_stack(shape=sample_shape)
        return xs

    def parameters(self):
        return list(self.flow_stack.parameters()) + list(filter(lambda x: x.requires_grad, self.flow_stack.base_args.values()))

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
            self.flow_stack.base_dist(**self.flow_stack.base_args)
            .log_prob(z).squeeze()
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
        num_flows: int = 8,
        lr: float = 1e-2,
        base_dist: Optional[dist.Distribution] = dist.Normal,
        base_args = {'loc': torch.tensor(0.0), 'scale': torch.tensor(1.0)},
    ) -> Dict[RVIdentifier, VariationalApproximation]:
        try:
            random_seed = (
                torch.randint(MeanFieldVariationalInference._rand_int_max, (1,)).int().item()
            )
            MeanFieldVariationalInference.set_seed(random_seed, 0)
            self.queries_ = queries
            self.observations_ = observations

            # TODO: handle dimension
            vi_dicts = defaultdict(lambda: VariationalApproximation(
                num_flows=num_flows, 
                lr=lr, 
                base_dist=base_dist,
                base_args=base_args))
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

                nodes = list(self.world_.get_all_world_vars().items())
                shuffle(nodes)
                for rvid, node_var in nodes:
                    if rvid in self.observations_:
                        continue
                    else:
                        proposer = vi_dicts[rvid]

                        def _target_log_prob(x):
                            self.world_.propose_change(
                                rvid, x, start_new_diff=True
                            )
                            # sum here to prevent broadcasting of child log_prob
                            log_prob = node_var.distribution.log_prob(x).sum()
                            for child in node_var.children:
                                child_var = self.world_.get_node_in_world_raise_error(child)
                                log_prob += child_var.log_prob
                            self.world_.reject_diff()
                            return log_prob
                        proposer.target_log_prob = _target_log_prob
                        proposer.train(epochs=1)

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