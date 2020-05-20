# Copyright (c) Facebook, Inc. and its affiliates.
import warnings
from functools import wraps

from beanmachine.ppl.model.utils import Mode, RVIdentifier
from beanmachine.ppl.world.world import World


class StatisticalModel(object):
    """
    Parent class to all statistical models implemented in Bean Machine.

    every random variable in the model needs to be defined with function
    declaration accompanied with @sample decorator.

    for instance, here is Gaussian Mixture Model implementation:


    K, alpha, beta, gamma = init()

    @sample
    def mu():
        return Normal(alpha, beta)

    @sample
    def z(i):
        return Uniform(K)

    @sample
    def y(i):
        return Normal(mu(z(i)), gamma)
    """

    __world_ = None
    __mode_ = Mode.INITIALIZE

    @staticmethod
    def reset():
        """
        Initialize world at the beginning of inference
        """
        StatisticalModel.__world_ = World()
        StatisticalModel.__mode_ = Mode.INITIALIZE
        return StatisticalModel.__world_

    @staticmethod
    def get_world() -> World:
        """
        :returns: __world_
        """
        return StatisticalModel.__world_

    @staticmethod
    def get_mode() -> Mode:
        """
        :returns: __mode_
        """
        return StatisticalModel.__mode_

    @staticmethod
    def set_mode(mode):
        """
        Update __mode_ to mode

        :param mode: the mode to update the __mode_ to
        """
        StatisticalModel.__mode_ = mode

    @staticmethod
    def get_func_key(function, arguments) -> RVIdentifier:
        """
        Creates a key to uniquely identify the Random Variable.

        :param function: reference to function
        :param arguments: function arguments

        :returns: tuple of function and arguments which is to be used to identify
        a particular function call.
        """
        return RVIdentifier(function=function, arguments=arguments)

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
            func_key = StatisticalModel.get_func_key(f, args)
            if StatisticalModel.__mode_ == Mode.INITIALIZE:
                return func_key
            world = StatisticalModel.__world_
            return world.update_graph(func_key)

        f._wrapper = wrapper
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
            if StatisticalModel.__mode_ == Mode.INITIALIZE:
                return StatisticalModel.get_func_key(f, args)

            return f(*args)

        f._wrapper = wrapper
        return wrapper


random_variable = StatisticalModel.random_variable
sample = random_variable

functional = StatisticalModel.functional
query = functional
