import copy
import logging
from abc import ABCMeta
from functools import lru_cache
from typing import Callable, Dict, List, Optional

import torch
import torch.distributions as dist
import torch.optim
from torch import Tensor
from tqdm.auto import tqdm

from ...inference.abstract_infer import AbstractInference
from ...model.rv_identifier import RVIdentifier
from ...model.utils import LogLevel
from .variational_approximation import (
    ContinuousVariationalApproximation,
    DiscreteVariationalApproximation,
)


LOGGER = logging.getLogger("beanmachine.vi")


class MeanFieldVariationalInference(AbstractInference, metaclass=ABCMeta):
    """Inference class for mean-field variational inference.

    Fits a mean-field reparameterized guide on unconstrained latent space
    following ADVI (https://arxiv.org/pdf/1603.00788.pdf). The mean-field
    factors are IAF transforms (https://arxiv.org/pdf/1606.04934.pdf) of a
    given `base_dist`.
    """

    def _vi_approx_for_rv(self, rvid, base_dist, base_args, lr):
        node_var = self.world_.get_node_in_world_raise_error(rvid)
        if node_var.distribution.has_enumerate_support:
            return DiscreteVariationalApproximation(
                lr=lr,
                enumerated_support=node_var.distribution.enumerate_support(),
            )
        return ContinuousVariationalApproximation(
            lr=lr,
            event_shape=node_var.distribution.event_shape,
            base_dist=base_dist,
            base_args=copy.deepcopy(base_args),
        )

    def _target_log_prob(self, rvid, node_var, x):
        "Form single-site Gibbs log density x_s -> p(x_s, x_\\s)"
        # TODO: support batching on `x` (e.g. if `children` depend
        # on the value of `x`)
        self.world_.propose_change(rvid, x, start_new_diff=True)
        log_prob = node_var.distribution.log_prob(x).sum()
        for child in node_var.children:
            child_var = self.world_.get_node_in_world_raise_error(child)
            log_prob += child_var.log_prob
        self.world_.reject_diff()
        return log_prob

    def infer(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, Tensor],
        num_iter: int = 100,
        lr: float = 1e-2,
        base_dist: Optional[dist.Distribution] = None,
        base_args: Optional[dict] = None,
        random_seed: Optional[int] = None,
        num_elbo_mc_samples: int = 100,
        on_iter: Optional[Callable] = None,
    ) -> Callable[[RVIdentifier], ContinuousVariationalApproximation]:
        """
        Trains a set of mean-field variational approximation (one per site).

        :param queries: queried random variables
        :param observations: observations dict
        :param num_iter: number of  worlds to train over
        :param lr: learning rate
        :param base_dist: constructor fn for base distribution for flow
        :param base_args: arguments to base_dist (will optimize any `nn.Parameter`s)
        """
        if not base_dist:
            base_dist = dist.Normal
            base_args = {"loc": torch.tensor(0.0), "scale": torch.tensor(1.0)}
        if not base_args:
            base_args = {}
        try:
            if not random_seed:
                random_seed = (
                    torch.randint(MeanFieldVariationalInference._rand_int_max, (1,))
                    .int()
                    .item()
                )
            self.set_seed(random_seed)
            self.queries_ = queries
            self.observations_ = observations

            vi_dicts = lru_cache(maxsize=None)(
                lambda rvid: self._vi_approx_for_rv(rvid, base_dist, base_args, lr)
            )
            # pyre-fixme[16]: Module `tqdm.auto` has no attribute `tqdm`.
            for iter_num in tqdm(iterable=range(num_iter), desc="Training iterations"):
                # sample world x ~ q_t
                self.initialize_world(False, vi_dicts)

                nodes = self.world_.get_all_world_vars()
                latent_rvids = list(
                    filter(lambda rvid: rvid not in self.observations_, nodes.keys())
                )
                loss = torch.zeros(1)
                # iterate over latent sites x_s.
                for rvid in latent_rvids:
                    node_var = nodes[rvid]
                    v_approx = vi_dicts(rvid)

                    # decompose mean-field ELBO expectation E_x = E_\s E_s and
                    # MC approximate E_\s using previously sampled x_\s, i.e.
                    # ELBO = E_x log p/q ~= E_s log p(x_s, x_\s) / q(x_s)

                    # MC approximate E_s using `num_elbo_mc_samples` (reparameterized)
                    # samples x_{s,i} ~ q_t(x_s) i.e.
                    # ELBO ~= E_s log p(x_s, x_\s) / q(x_s)
                    #      ~= (1/N) \sum_i^N log p(x_{s,i}, x_\s) / q(x_{s,i})
                    if node_var.distribution.has_enumerate_support:  # pyre-fixme[16]
                        loss -= v_approx.elbo(
                            target_log_prob=lambda x: self._target_log_prob(
                                rvid, node_var, x
                            )
                        )
                    else:
                        loss -= v_approx.elbo(
                            target_log_prob=lambda x: self._target_log_prob(
                                rvid, node_var, x
                            ),
                            target_support=node_var.distribution.support,  # pyre-fixme[16]
                            num_elbo_mc_samples=num_elbo_mc_samples,
                        )

                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward(retain_graph=True)
                    for rvid in latent_rvids:
                        v_approx = vi_dicts(rvid)
                        v_approx.optim.step()
                        v_approx.optim.zero_grad()
                else:
                    # TODO: caused by e.g. negative scales in `dist.Normal`;
                    # fix using pytorch's `constraint_registry` to account for
                    # `Distribution.arg_constraints`
                    LOGGER.log(
                        LogLevel.INFO.value, "Encountered NaNs in loss, skipping epoch"
                    )

                if on_iter:
                    on_iter(iter_num, loss)

        except BaseException as x:
            raise x
        finally:
            self.reset()
        return vi_dicts
