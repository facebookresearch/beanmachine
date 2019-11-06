# Copyright (c) Facebook, Inc. and its affiliates.
from collections import defaultdict
from functools import wraps
from typing import Dict, List

from beanmachine.ppl.model.utils import Mode, RVIdentifier
from beanmachine.ppl.world.variable import Variable
from beanmachine.ppl.world.world import World
from torch import Tensor


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

    __stack_ = []
    __world_ = None
    __mode_ = Mode.INITIALIZE
    __observe_vals_ = defaultdict()

    @staticmethod
    def reset():
        """
        Initialize world and stack at the beginning of inference
        """
        StatisticalModel.__stack_ = []
        StatisticalModel.__world_ = World()
        StatisticalModel.__mode_ = Mode.INITIALIZE
        StatisticalModel.__observe_vals_ = defaultdict()
        return StatisticalModel.__stack_, StatisticalModel.__world_

    @staticmethod
    def get_stack() -> List[RVIdentifier]:
        """
        :returns: __stack_
        """
        # pyre-fixme[16]: `Type` has no attribute `__stack_`.
        return StatisticalModel.__stack_

    @staticmethod
    def get_world() -> World:
        """
        :returns: __world_
        """
        # pyre-fixme[16]: `Type` has no attribute `__world_`.
        # pyre-fixme[7]: Expected `World` but got `None`.
        return StatisticalModel.__world_

    @staticmethod
    def get_mode() -> Mode:
        """
        :returns: __mode_
        """
        # pyre-fixme[16]: `Type` has no attribute `__mode_`.
        return StatisticalModel.__mode_

    @staticmethod
    def set_mode(mode):
        """
        Update __mode_ to mode

        :param mode: the mode to update the __mode_ to
        """
        StatisticalModel.__mode_ = mode

    @staticmethod
    def get_observations() -> Dict[RVIdentifier, Tensor]:
        """
        :returns: __observe_vals_
        """
        # pyre-fixme[16]: `Type` has no attribute `__observe_vals_`.
        return StatisticalModel.__observe_vals_

    @staticmethod
    def set_observations(val):
        """
        Update __observe_vals_ to val

        :param val: the value to set the __observe_vals_ to
        """
        StatisticalModel.__world_.set_observations(val)
        StatisticalModel.__observe_vals_ = val

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
            stack = StatisticalModel.__stack_
            obs = StatisticalModel.__observe_vals_

            if len(stack) > 0:
                world.get_node_in_world(stack[-1]).parent.add(func_key)

            var = world.get_node_in_world(func_key)
            if var is not None:
                if len(stack) > 0:
                    var.children.add(stack[-1])

                return var.value

            var = Variable(
                distribution=None,
                value=None,
                log_prob=None,
                parent=set(),
                children=set() if len(stack) == 0 else set({stack[-1]}),
                proposal_distribution=None,
                extended_val=None,
            )

            world.add_node_to_world(func_key, var)
            stack.append(func_key)
            distribution = f(*args)
            stack.pop()

            var.distribution = distribution
            obs = obs[func_key] if func_key in obs else None
            var.initialize_value(obs)

            var.log_prob = distribution.log_prob(var.value).sum()
            world.update_diff_log_prob(func_key)
            return var.value

        f._wrapper = wrapper
        return wrapper

    @staticmethod
    def query(f):
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


sample = StatisticalModel.sample
query = StatisticalModel.query
