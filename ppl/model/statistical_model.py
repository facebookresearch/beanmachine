# Copyright (c) Facebook, Inc. and its affiliates.


class StatisticalModel:
    """
    parent class to all statistical models implemented in Bean Machine.

    every random variable in the model needs to be defined with function
    declaration accompanied with @sample decorator.

    for instance, here is Gaussian Mixture Model implementation:

    class Model(StatisticalModel):
        def __init__(self, C, K, alpha=0, beta=100, gamma=10):
            self.K, self.alpha, self.beta, self.gamma = K, alpha, beta, gamma

            @sample
            def mu(self):
                return Normal(self.alpha, self.beta)

            @sample
            def z(self, i):
                return Uniform(self.K)

            @sample
            def y(self, i):
                return Normal(self.mu(self.z(i)), self.gamma)
    """

    @staticmethod
    def get_func_key(name, arg):
        return (name, arg)
