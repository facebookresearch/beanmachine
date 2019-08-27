# Copyright (c) Facebook, Inc. and its affiliates.
from collections import defaultdict, namedtuple
from functools import wraps

from beanmachine.ppl.model.utils import Mode
from beanmachine.ppl.world.variable import Variable
from beanmachine.ppl.world.world import World


RandomVariable = namedtuple("RandomVariable", "function arguments")


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
    def initialize():
        """
        Initialize world and stack at the beginning of inference
        """
        StatisticalModel.__stack_ = []
        StatisticalModel.__world_ = World()
        StatisticalModel.__mode_ = Mode.INITIALIZE
        return StatisticalModel.__stack_, StatisticalModel.__world_

    @staticmethod
    def get_stack():
        """
        Returns __stack_
        """
        return StatisticalModel.__stack_

    @staticmethod
    def get_world():
        """
        Returns __world_
        """
        return StatisticalModel.__world_

    @staticmethod
    def get_mode():
        """
        Returns __mode_
        """
        return StatisticalModel.__mode_

    @staticmethod
    def set_mode(mode):
        """
        Update __mode_ to mode
        """
        StatisticalModel.__mode_ = mode

    @staticmethod
    def get_observations():
        """
        Returns __observe_vals_
        """
        return StatisticalModel.__observe_vals_

    @staticmethod
    def set_observations(val):
        """
        Update __observe_vals_ to val
        """
        StatisticalModel.__observe_vals_ = val

    @staticmethod
    def get_func_key(name, args):
        """
        Creates a function signature.

        parameters are:
            name: function name
            args: function arguments

        returns:
            tuple of name and argument which is to be used as function signature
        """
        return RandomVariable(function=name, arguments=args)

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
            )

            world.add_node_to_world(func_key, var)
            stack.append(func_key)
            distribution = f(*args)
            stack.pop()

            var.distribution = distribution
            var.value = obs[func_key] if func_key in obs else distribution.sample()
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
