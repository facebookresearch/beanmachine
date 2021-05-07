# Copyright (c) Facebook, Inc. and its affiliates.
import warnings
from functools import wraps

from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world import World, get_world_context


class StatisticalModel(object):
    """
    Parent class to all statistical models implemented in Bean Machine.

    every random variable in the model needs to be defined with function
    declaration accompanied with @bm.random_variable decorator.

    for instance, here is Gaussian Mixture Model implementation:


    K, alpha, beta, gamma = init()

    @bm.random_variable
    def mu():
        return Normal(alpha, beta)

    @bm.random_variable
    def z(i):
        return Uniform(K)

    @bm.random_variable
    def y(i):
        return Normal(mu(z(i)), gamma)
    """

    @staticmethod
    def get_func_key(wrapper, arguments) -> RVIdentifier:
        """
        Creates a key to uniquely identify the Random Variable.

        :param wrapper: reference to the wrapper function
        :param arguments: function arguments

        :returns: tuple of function and arguments which is to be used to identify
        a particular function call.
        """
        return RVIdentifier(wrapper=wrapper, arguments=arguments)

    @staticmethod
    def sample(f):
        warnings.warn(
            "@sample will be deprecated, use @random_variable instead",
            DeprecationWarning,
        )
        return StatisticalModel.random_variable(f)

    @staticmethod
    def random_variable(f):
        """
        Decorator to be used for every stochastic random variable defined in
        all statistical models.
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
    def query(f):
        warnings.warn(
            "@query will be deprecated, use @functional instead", DeprecationWarning
        )
        return StatisticalModel.functional(f)

    @staticmethod
    def functional(f):
        """
        Decorator to be used for every query defined in statistical model.
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
        Decorator to be used for params (variable to be optimized).

        TODO: DRY out with `random_variable`
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
sample = random_variable

functional = StatisticalModel.functional
query = functional

param = StatisticalModel.param
