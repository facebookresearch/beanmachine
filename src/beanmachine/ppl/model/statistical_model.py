# Copyright (c) Facebook, Inc. and its affiliates.
import warnings

from beanmachine.ppl.model import RVWrapper


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
        return RVWrapper(function=f)

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
        return RVWrapper(function=f, is_random_variable=False)


random_variable = StatisticalModel.random_variable
sample = random_variable

functional = StatisticalModel.functional
query = functional
