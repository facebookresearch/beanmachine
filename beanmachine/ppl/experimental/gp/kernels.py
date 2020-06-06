import gpytorch.kernels as kernels
from beanmachine.ppl.model.utils import RVIdentifier


class KernelMixin(object):
    """
    Wrapper for Gpytorch kernels that overrides the parameter attributes
    with invocations of a Bean Machine `random_variable`::

       @random_variable
       def prior():
           return dist.Normal(...)

       x = ScaleKernel(outputscale_prior=prior)
       x.lengthscale  # returns an invocation of the prior function
                      # (used during inference)

    """

    def _validate_args(self, prior):
        assert isinstance(prior(), RVIdentifier)
        "Prior should be None or a random variable but was: {}".format(type(prior))

    def __init__(self, **kwargs):
        self.priors = {}
        for k, v in kwargs.copy().items():
            if "prior" not in k:
                continue

            self._validate_args(v)
            self.priors[k] = v

            # remove the prior for GPytorch
            kwargs.pop(k)

        super().__init__(**kwargs)

    @property
    def lengthscale(self):
        if "lengthscale_prior" in self.priors:
            return self.priors["lengthscale_prior"]()
        return super().lengthscale

    @property
    def outputscale(self):
        if "outputscale_prior" in self.priors:
            return self.priors["outputscale_prior"]()
        return super().outputscale

    @property
    def variance(self):
        if "variance_prior" in self.priors:
            return self.priors["variance_prior"]()
        return super().variance

    @property
    def offset(self):
        if "offset_prior" in self.priors:
            return self.priors["offset_prior"]()
        return super().offset

    @property
    def angular_weights(self):
        if "angular_weights_prior" in self.priors:
            return self.priors["angular_weights_prior"]()
        return super().angular_weights

    @property
    def alpha(self):
        if "alpha_prior" in self.priors:
            return self.priors["alpha_prior"]()
        return super().alpha

    @property
    def beta(self):
        if "beta_prior" in self.priors:
            return self.priors["beta_prior"]()
        return super().beta

    @property
    def mixture_means(self):
        if "mixture_means_prior" in self.priors:
            return self.priors["mixture_means_prior"]()
        return super().mixture_means

    @property
    def mixture_scales(self):
        if "mixture_scales_prior" in self.priors:
            return self.priors["mixture_scales_prior"]()
        return super().mixture_scales

    @property
    def mixture_weights(self):
        if "mixture_weights_prior" in self.priors:
            return self.priors["mixture_weights_prior"]()
        return super().mixture_weights

    @property
    def task_covar(self):
        if "task_covar_prior" in self.priors:
            return self.priors["task_covar_prior"]()
        return super().task_covar

    @property
    def period_length(self):
        if "period_length_prior" in self.priors:
            return self.priors["period_length_prior"]()
        return super().period_length


all_kernels = []
# Wrap all the kernels from GPytorch
for name, kernel in kernels.__dict__.items():
    if not isinstance(kernel, type):
        continue
    if not issubclass(kernel, kernels.Kernel):
        continue

    all_kernels.append(name)

    bm_kernel = type(name, (KernelMixin, kernel), {})
    bm_kernel.__module__ = __name__
    bm_kernel.__doc__ = """
    Wraps `{}`.{}` with `beanmachine.ppl.experimental.kernels.KernelMixin.
    """.format(
        bm_kernel.__module__, bm_kernel.__name__
    )
    locals()[name] = bm_kernel
