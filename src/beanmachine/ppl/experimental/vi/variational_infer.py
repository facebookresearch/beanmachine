# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from abc import ABCMeta
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.optim
from torch import Tensor
from tqdm.auto import tqdm

from ...legacy.inference.abstract_infer import AbstractInference
from ...model.rv_identifier import RVIdentifier
from ...model.utils import LogLevel
from .mean_field_variational_approximation import MeanFieldVariationalApproximation
from .optim import BMMultiOptimizer, BMOptim


LOGGER = logging.getLogger("beanmachine.vi")
cpu_device = torch.device("cpu")
default_params = {}


class MeanFieldVariationalInference(AbstractInference, metaclass=ABCMeta):
    """Inference class for mean-field variational inference.

    Fits a mean-field reparameterized guide on unconstrained latent space
    following ADVI (https://arxiv.org/pdf/1603.00788.pdf). The mean-field
    factors are IAF transforms (https://arxiv.org/pdf/1606.04934.pdf) of a
    given `base_dist`.
    """

    def infer(  # noqa: C901
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, Tensor],
        num_iter: int = 100,
        lr: float = 1e-3,
        base_dist: Optional[dist.Distribution] = None,
        base_args: Optional[dict] = None,
        random_seed: Optional[int] = None,
        num_elbo_mc_samples=100,
    ) -> Callable[[RVIdentifier], MeanFieldVariationalApproximation]:
        """
        Trains a set of mean-field variational approximation (one per site).

        All tensors in `queries` and `observations` must be allocated on the
        same `torch.device`. Inference algorithms will attempt to allocate
        intermediate tensors on the same device.

        :param queries: queried random variables
        :param observations: observations dict
        :param num_iter: number of  worlds to train over
        :param lr: learning rate
        :param base_dist: constructor fn for base distribution for flow
        :param base_args: arguments to base_dist (will optimize any `nn.Parameter`s)
        """
        if not base_dist:
            base_dist = dist.Normal
            base_args = {"loc": torch.tensor([0.0]), "scale": torch.tensor([1.0])}
            # TODO: reinterpret batch dimension?
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

            def _get_var_approx(rvid):
                target_dist = self.world_.get_node_in_world_raise_error(
                    rvid
                ).distribution
                # NOTE: this assumes the target distribution's event_shape and
                # support do not change
                return MeanFieldVariationalApproximation(
                    lr=lr,
                    target_dist=target_dist,
                    base_dist=base_dist,
                    base_args=copy.deepcopy(base_args),
                )

            vi_dicts = lru_cache(maxsize=None)(_get_var_approx)
            for _ in tqdm(iterable=range(num_iter), desc="Training iterations"):
                # sample world x ~ q_t
                self.initialize_world(False, vi_dicts)

                nodes = self.world_.get_all_world_vars()
                latent_rvids = list(
                    filter(lambda rvid: rvid not in self.observations_, nodes.keys())
                )
                loss = torch.zeros(1)
                # decompose mean-field ELBO expectation E_x = E_s E_\s and
                # iterate over latent sites x_s.
                for rvid in latent_rvids:
                    v_approx = vi_dicts(rvid)

                    # Form single-site Gibbs density approximating E_\s using
                    # previously sampled x_\s, i.e. x_s -> E_\s log p(x_s, x_\s)
                    # ~= x_s -> log p(x_s, z) with z ~ p(x_\s)
                    def _target_log_prob(x):
                        (
                            _,
                            _,
                            _,
                            proposed_score,
                        ) = self.world_.propose_change_transformed_value(
                            rvid, x, start_new_diff=False
                        )
                        self.world_.reject_diff()
                        return proposed_score

                    # MC approximate E_s using `num_elbo_mc_samples` (reparameterized)
                    # samples x_{s,i} ~ q_t(x_s) i.e.
                    # ELBO ~= E_s log p(x_s, x_\s) / q(x_s)
                    #      ~= (1/N) \sum_i^N log p(x_{s,i}, x_\s) / q(x_{s,i})
                    loss -= v_approx.elbo(
                        _target_log_prob,
                        num_elbo_mc_samples,
                    )

                if not torch.isnan(loss) and not torch.isinf(loss):
                    for rvid in latent_rvids:
                        v_approx = vi_dicts(rvid)
                        v_approx.optim.zero_grad()
                    loss.backward(retain_graph=True)
                    for rvid in latent_rvids:
                        v_approx = vi_dicts(rvid)
                        v_approx.optim.step()
                        v_approx.recompute_transformed_distribution()
                else:
                    # TODO: caused by e.g. negative scales in `dist.Normal`;
                    # fix using pytorch's `constraint_registry` to account for
                    # `Distribution.arg_constraints`
                    LOGGER.log(
                        LogLevel.INFO.value, "Encountered NaNs in loss, skipping epoch"
                    )

        except BaseException as x:
            raise x
        finally:
            self.reset()
        return vi_dicts


class VariationalInference(AbstractInference, metaclass=ABCMeta):
    """
    Stochastic Variational Inference.

    Fits a variational approximation represented as a guide program by
    Monte-Carlo approximating ELBO and optimizing over any `bm.param`s
    used in the guide.
    """

    def __init__(self):
        super().__init__()

    def infer(
        self,
        model_to_guide_ids: Dict[RVIdentifier, RVIdentifier],
        observations: Dict[RVIdentifier, Tensor],
        num_iter: int = 100,
        lr: float = 1e-3,
        random_seed: Optional[int] = None,
        on_iter: Optional[Callable] = None,
        params: Dict[RVIdentifier, nn.Parameter] = default_params,
        optimizer: Optional[BMMultiOptimizer] = None,
        progress_bar: Optional[bool] = True,
        device: Optional[torch.device] = cpu_device,
    ) -> Dict[RVIdentifier, nn.Parameter]:
        """
        A multiple-step version of `.step()` to perform Stochastic Variational Inference.
        This is convenient for full-batch training.

        :param model_to_guide_ids: mapping from latent variables to their
            respective guide random variables
        :param observations: observed random variables with their values
        :param num_iter: number of iterations of optimizer steps
        :param lr: learning rate
        :param random_seed: random seed
        :param on_iter: callable executed after each optimizer iteration
        :param params: parameter random_variable keys and their values, used
            to initialize optimization if present
        :param optimizer: BMOptim (wrapped torch optimizer) instance to reuse
        :param progress_bar: flag for tqdm progress, disable for less output
            when minibatching
        :param device: default torch device for tensor allocations

        :returns: mapping from all `bm.param` `RVIdentifier`s encountered
            to their optimized values
        """
        try:
            if not random_seed:
                random_seed = (
                    torch.randint(MeanFieldVariationalInference._rand_int_max, (1,))
                    .int()
                    .item()
                )
            self.set_seed(random_seed)
            if not optimizer:
                # initialize world so guide params available
                self.queries_ = list(model_to_guide_ids.keys())
                self.observations_ = observations
                self.initialize_world(
                    False,
                    model_to_guide_ids=model_to_guide_ids,
                    params=params,
                )

                # optimizer = torch.optim.Adam(self.world_.params_.values(), lr=lr)
                optimizer = BMMultiOptimizer(
                    BMOptim(
                        torch.optim.Adam,
                        {"lr": lr},
                    )
                )

            for it in (
                tqdm(iterable=range(num_iter), desc="Training iterations")
                if progress_bar
                else range(num_iter)
            ):
                loss, params, optimizer = self.step(
                    model_to_guide_ids, observations, optimizer, params, device
                )
                if on_iter:
                    on_iter(it, loss, params)
        except BaseException as x:
            raise x
        finally:
            self.reset()
        return params

    def step(
        self,
        model_to_guide_ids: Dict[RVIdentifier, RVIdentifier],
        observations: Dict[RVIdentifier, Tensor],
        optimizer: BMMultiOptimizer,
        params: Dict[RVIdentifier, nn.Parameter] = default_params,
        device: Optional[torch.device] = cpu_device,
    ) -> Tuple[torch.Tensor, Dict[RVIdentifier, nn.Parameter], BMMultiOptimizer]:
        """
        Perform one step of stochastic variational inference.

        All `bm.param`s referenced in guide random variables are optimized to
        minimize a Monte Carlo approximation of negative ELBO loss. The
        negative ELBO loss is Monte-Carlo approximated by sampling the guide
        `bm.random_variable`s (i.e. values of `model_to_guide_ids`) to draw
        trace samples from the variational approximation and scored against
        the model `bm.random_variables` (i.e. keys of `model_to_guide_ids`).
        It is the end-user's responsibility to interpret the optimized values
        for the `bm.param`s returned.

        :param model_to_guide_ids: mapping from latent variables to their
        respective guide random variables
        :param observations: observed random variables with their values
        :param lr: learning rate
        :param random_seed: random seed
        :param params: parameter random_variable keys and their values, used
        to initialize optimization if present
        :param optimizer: optimizer state (e.g. momentum and weight decay)
        to reuse
        :param device: default torch device for tensor allocations

        :returns: loss value, mapping from all `bm.param` `RVIdentifier`s
        encountered to their optimized values, optimizer for the respective
        `bm.param` tensors
        """
        self.queries_ = list(model_to_guide_ids.keys())
        self.observations_ = observations

        # sample world x ~ q_t
        self.initialize_world(
            False,
            model_to_guide_ids=model_to_guide_ids,
            params=params,
        )

        # TODO: add new `self.world_.params_` not already in optimizer

        nodes = self.world_.get_all_world_vars()
        latent_rvids = list(
            filter(
                lambda rvid: (
                    rvid not in self.observations_
                    and rvid not in model_to_guide_ids.values()
                ),
                nodes.keys(),
            )
        )
        loss = torch.zeros(1).to(device)
        # -ELBO == E[log p(obs, x) - log q(x)] ~= log p(obs | x) +
        # \sum_s (log p(x_s) - log q(x_s)) where x_s ~ q_t were sampled
        # during `initialize_world`.
        # Here we compute the second term suming over latent sites x_s.
        for rvid in latent_rvids:
            assert (
                rvid in model_to_guide_ids
            ), f"expected every latent to have a guide, but did not find one for {rvid}"
            node_var = nodes[rvid]
            v_approx = nodes[model_to_guide_ids[rvid]]

            if isinstance(node_var.distribution, dist.Bernoulli) and isinstance(
                v_approx.distribution, dist.Bernoulli
            ):
                # binary cross entropy, analytical ELBO
                # TODO: more general enumeration
                loss += nn.BCELoss()(
                    # pyre-fixme[16]: `Distribution` has no attribute `probs`.
                    v_approx.distribution.probs,
                    node_var.distribution.probs,
                )

                # TODO: downstream observation likelihoods p(obs | rvid)
            else:
                # MC ELBO
                loss += v_approx.distribution.log_prob(node_var.value).sum()
                loss -= node_var.distribution.log_prob(node_var.value).sum()

                # Add the remaining likelihood term log p(obs | x)
                for obs_rvid in self.observations_:
                    obs_var = nodes[obs_rvid]
                    loss -= obs_var.distribution.log_prob(
                        self.observations_[obs_rvid]
                    ).sum()

        if not torch.isnan(loss):
            # loss.backward()
            optimizer.step(loss, self.world_.params_)
            # optimizer.zero_grad()
            params = self.world_.params_
        else:
            # TODO: caused by e.g. negative scales in `dist.Normal`;
            # fix using pytorch's `constraint_registry` to account for
            # `Distribution.arg_constraints`
            LOGGER.log(LogLevel.INFO.value, "Encountered NaNs in loss, skipping epoch")

        return loss, params, optimizer
