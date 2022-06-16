# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABCMeta, abstractmethod
from typing import Iterable

import torch
from beanmachine import ppl as bm
from beanmachine.ppl.distributions.delta import Delta
from beanmachine.ppl.experimental.vi.variational_infer import VariationalInfer
from beanmachine.ppl.experimental.vi.variational_world import VariationalWorld
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world import RVDict
from torch import distributions as dist
from torch.distributions.constraint_registry import biject_to
from torch.nn.functional import softplus


class AutoGuideVI(VariationalInfer, metaclass=ABCMeta):
    """VI with guide distributions automatically generated."""

    def __init__(
        self,
        queries: Iterable[RVIdentifier],
        observations: RVDict,
        **kwargs,
    ):
        queries_to_guides = {}

        # runs all queries to discover their dimensions
        self._world = VariationalWorld(
            observations=observations,
            params={},
            queries_to_guides=queries_to_guides,
        )
        # automatically instantiate `queries_to_guides`
        for query in queries:
            self._world.call(query)
            distrib = self._world.get_variable(query).distribution
            queries_to_guides[query] = self.get_guide(query, distrib)

        super().__init__(
            queries_to_guides=queries_to_guides,
            observations=observations,
            **kwargs,
        )

    @staticmethod
    @abstractmethod
    def get_guide(query, distrib) -> RVIdentifier:
        pass


class ADVI(AutoGuideVI):
    """Automatic Differentiation Variational Inference (ADVI).

    ADVI automates construction of guides by initializing variational distributions
    as Gaussians and possibly bijecting them so the supports match.

    See https://arxiv.org/abs/1506.03431.
    """

    @staticmethod
    def get_guide(query, distrib):
        @bm.param
        def param_loc():
            # TODO: use event shape
            return (
                torch.rand_like(biject_to(distrib.support).inv(distrib.sample())) * 4.0
                - 2.0
            )

        @bm.param
        def param_scale():
            # TODO: use event shape
            return (
                0.01
                + torch.rand_like(biject_to(distrib.support).inv(distrib.sample()))
                * 4.0
                - 2.0
            )

        def f():
            loc = param_loc()
            scale = softplus(param_scale())
            q = dist.Normal(loc, scale)
            if distrib.support != dist.constraints.real:
                if distrib.support == dist.constraints.positive:
                    # override exp transform with softplus
                    q = dist.TransformedDistribution(
                        q, [dist.transforms.SoftplusTransform()]
                    )
                else:
                    q = dist.TransformedDistribution(q, [biject_to(distrib.support)])
            return q

        f.__name__ = "guide_" + str(query)
        return bm.random_variable(f)()


class MAP(AutoGuideVI):
    """Maximum A Posteriori (MAP) Inference.

    Uses ``Delta`` distributions to perform a point estimate
    of the posterior mode.
    """

    @staticmethod
    def get_guide(query, distrib):
        @bm.param
        def param_loc():
            # TODO: use event shape
            return (
                torch.rand_like(biject_to(distrib.support).inv(distrib.sample())) * 4.0
                - 2.0
            )

        def f():
            loc = param_loc()
            if distrib.support != dist.constraints.real:
                if distrib.support == dist.constraints.positive:
                    loc = dist.transforms.SoftplusTransform()(loc)
                else:
                    loc = biject_to(distrib.support)(loc)
            q = Delta(loc)
            return q

        f.__name__ = "guide_" + str(query)
        return bm.random_variable(f)()
