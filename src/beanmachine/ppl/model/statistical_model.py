# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import wraps

from beanmachine.ppl.legacy.world import World
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world import get_world_context


class StatisticalModel(object):
    """
    Parent class to all statistical models implemented in Bean Machine.

    Every random variable in the model needs to be defined with function
    declaration wrapped the ``bm.random_variable`` .

    Every deterministic functional that a user would like to query during
    inference should be wrapped in a ``bm.functional`` .

    [EXPERIMENTAL]: Every parameter of the guide distribution that is to be learned
    via variational inference should be wrapped in a ``bm.param`` .
    """

    @staticmethod
    def get_func_key(wrapper, arguments) -> RVIdentifier:
        """
        Creates a key to uniquely identify the Random Variable.

        Args:
          wrapper: reference to the wrapper function
          arguments: function arguments

        Returns:
          Tuple of function and arguments which is to be used to identify
          a particular function call.
        """
        return RVIdentifier(wrapper=wrapper, arguments=arguments)

    @staticmethod
    def random_variable(f):
        """
        Decorator to be used for every stochastic random variable defined in
        all statistical models. E.g.::

          @bm.random_variable
          def foo():
            return Normal(0., 1.)

          def foo():
            return Normal(0., 1.)
          foo = bm.random_variable(foo)
        """

        @wraps(f)
        def wrapper(*args):
            func_key = StatisticalModel.get_func_key(wrapper, args)
            world = get_world_context()
            if world is None:
                return func_key
            else:
                return world.update_graph(func_key)

        wrapper.is_functional = False
        wrapper.is_random_variable = True
        return wrapper

    @staticmethod
    def functional(f):
        """
        Decorator to be used for every query defined in statistical model, which are
        functions of ``bm.random_variable`` ::

          @bm.random_variable
          def foo():
            return Normal(0., 1.)

          @bm.functional():
          def bar():
            return foo() * 2.0
        """

        @wraps(f)
        def wrapper(*args):
            world = get_world_context()
            if world is None:
                return StatisticalModel.get_func_key(wrapper, args)
            elif isinstance(world, World) and world.get_cache_functionals():
                return world.update_cached_functionals(f, *args)
            else:
                return f(*args)

        wrapper.is_functional = True
        wrapper.is_random_variable = False
        return wrapper

    @staticmethod
    def param(init_fn):
        """
        Decorator to be used for params (variable to be optimized with VI).::

          @bm.param
          def mu():
            return Normal(0., 1.)

          @bm.random_variable
          def foo():
            return Normal(mu(), 1.)
        """

        @wraps(init_fn)
        def wrapper(*args):
            func_key = StatisticalModel.get_func_key(wrapper, args)
            world = get_world_context()
            if isinstance(world, World):
                return world.get_param(func_key)
            else:
                return func_key

        return wrapper


random_variable = StatisticalModel.random_variable
functional = StatisticalModel.functional
param = StatisticalModel.param
