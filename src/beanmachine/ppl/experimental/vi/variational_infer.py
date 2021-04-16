import copy
import logging
from abc import ABCMeta
from functools import lru_cache
from typing import Callable, Dict, List, Optional

import flowtorch
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.optim
from torch import Tensor
from tqdm.auto import tqdm

from ...inference.abstract_infer import AbstractInference
from ...model.rv_identifier import RVIdentifier
from ...model.utils import LogLevel
from .mean_field_variational_approximation import MeanFieldVariationalApproximation


LOGGER = logging.getLogger("beanmachine.vi")


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
        flow: Optional[Callable[[], flowtorch.Bijector]] = None,
        base_dist: Optional[dist.Distribution] = None,
        base_args: Optional[dict] = None,
        random_seed: Optional[int] = None,
        num_elbo_mc_samples=100,
        on_iter: Optional[Callable] = None,
        vi_dicts: Optional[
            Callable[[RVIdentifier], MeanFieldVariationalApproximation]
        ] = None,
        pretrain: bool = False,
        progress_bar: bool = False,
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
            # default to mean-field ADVI
            base_dist = dist.Normal
            base_args = {
                "loc": nn.Parameter(torch.tensor([0.0])),
                "scale": nn.Parameter(torch.tensor([1.0])),
            }
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
                    flow=flow,
                    base_dist=base_dist,
                    base_args=copy.deepcopy(base_args),
                )

            vi_dicts = lru_cache(maxsize=None)(_get_var_approx)
            train_iter = range(num_iter)
            if progress_bar:
                train_iter = tqdm(iterable=train_iter, desc="Training iterations")
            for it in train_iter:
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
                    # Form single-site Gibbs log density x_s -> p(x_s, x_\s)
                    def _target_log_prob(x):
                        # TODO: support batching on `x` (e.g. if `children` depend
                        # on the value of `x`)
                        self.world_.propose_change(rvid, x, start_new_diff=True)
                        log_prob = node_var.distribution.log_prob(x).sum()
                        for child in node_var.children:
                            child_var = self.world_.get_node_in_world_raise_error(child)
                            log_prob += child_var.log_prob
                        self.world_.reject_diff()
                        return log_prob

                    # MC approximate E_s using `num_elbo_mc_samples` (reparameterized)
                    # samples x_{s,i} ~ q_t(x_s) i.e.
                    # ELBO ~= E_s log p(x_s, x_\s) / q(x_s)
                    #      ~= (1/N) \sum_i^N log p(x_{s,i}, x_\s) / q(x_{s,i})
                    prev_loss = loss.clone().detach()
                    delta_elbo, zk = v_approx.elbo(
                        _target_log_prob
                        if not pretrain
                        else node_var.distribution.log_prob,
                        num_elbo_mc_samples,
                    )
                    loss -= delta_elbo
                    # if (
                    #     str(rvid).startswith("sigma")
                    #     and delta_elbo.abs() > 1e2
                    #     and not pretrain
                    # ):
                    #     # v_approx._transform = dist.transforms.AbsTransform(
                    #     print("delta_elbo", delta_elbo)
                    #     print("zk", zk)
                    #     print("p(x,rest)", _target_log_prob(zk))
                    #     print("p(x|rest)", node_var.distribution.log_prob(zk))
                    #     print("q(x)", v_approx.log_prob(zk))
                    #     print(v_approx._transform)
                    #     print("=" * 10)

                if not torch.isnan(loss) and not torch.isinf(loss):
                    for rvid in latent_rvids:
                        v_approx = vi_dicts(rvid)
                        v_approx.optim.zero_grad()
                    loss.backward(retain_graph=True)
                    for rvid in latent_rvids:
                        v_approx = vi_dicts(rvid)
                        nn.utils.clip_grad_norm_(v_approx.parameters(), 5e3)
                        v_approx.optim.step()
                        v_approx.recompute_transformed_distribution()

                    if on_iter:
                        on_iter(it, loss, vi_dicts)
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
    ) -> Dict[RVIdentifier, Tensor]:
        """
        Perform stochastic variational inference.

        All `bm.param`s referenced in guide random variables are optimized to
        minimize a Monte Carlo approximation of negative ELBO loss. The
        negative ELBO loss is Monte-Carlo approximated by sampling the guide
        `bm.random_variable`s (i.e. values of `model_to_guide_ids`) to draw
        trace samples from the variational approximation and scored against
        the model `bm.random_variables` (i.e. keys of `model_to_guide_ids`).
        It is the end-user's responsibility to interpret the optimized values
        for the `bm.param`s returned.

        NOTE: only sequential SVI is supported; parallel is blocked until
        we consistently handle batch vs event dimensions within beanmachine.

        :param model_to_guide_ids: mapping from latent variables to their
        respective guide random variables
        :param observations: observed random variables with their values
        :param num_iter: number of iterations of optimizer steps
        :param lr: learning rate
        :param random_seed: random seed

        :returns: mapping from all `bm.param` `RVIdentifier`s encountered
        to their optimized values
        """
        optimizer = None

        try:
            if not random_seed:
                random_seed = (
                    torch.randint(MeanFieldVariationalInference._rand_int_max, (1,))
                    .int()
                    .item()
                )
            self.set_seed(random_seed)
            self.queries_ = list(model_to_guide_ids.keys())
            self.observations_ = observations

            params = {}

            for _ in tqdm(iterable=range(num_iter), desc="Training iterations"):
                # sample world x ~ q_t
                self.initialize_world(
                    False,
                    model_to_guide_ids=model_to_guide_ids,
                    params=params,
                )
                if not optimizer:
                    optimizer = torch.optim.Adam(self.world_.params_.values(), lr=lr)
                # TODO: add new guide params not already in optimizer

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
                loss = torch.zeros(1)
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

                    loss += v_approx.distribution.log_prob(node_var.value).sum()
                    loss -= node_var.distribution.log_prob(node_var.value).sum()

                # Add the remaining likelihood term log p(obs | x)
                for obs_rvid in self.observations_:
                    obs_var = nodes[obs_rvid]
                    loss -= obs_var.distribution.log_prob(
                        self.observations_[obs_rvid]
                    ).sum()

                if not torch.isnan(loss):
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    params = self.world_.params_
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
        return params
