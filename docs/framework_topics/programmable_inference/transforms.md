---
id: transforms
title: 'Transforms'
sidebar_label: 'Transforms'
slug: '/transforms'
---

Bean Machine provides flexibility for users to specify transformations on a per-variable basis. This gives Bean Machine powerful functionality.

Proposal algorithms will behave differently depending on the shape and constraints of the posterior, and often have specific requirements. It is useful to transform the posterior into an ideal shape and space for inference to be its most efficient. For example, the Hamiltonian Monte Carlo algorithm provided by Bean Machine requires the proposal distribution to be continuous and differentiable at all points in the real space. Therefore, for all variables with distributions constrained to subsets of the real numbers (such as non-negative ones, for example), HMC will require a transform to change the proposal distribution into the unconstrained space (for example, the transform `log(x)` will map points in the constrained space of all non-positive real numbers `x` into the unconstrained set of real numbers).

Bean Machine allows users to use default transformations for transforming constrained spaces into unconstrained spaces, or to specify custom transforms. Additionally, transforms can also be used in other ways such as specifying kernels for Gaussian processes.

Transforms are supported within the `Variable` class by the following attributes: `value`, `transformed_value` and `jacobian` (a generalization of derivatives for multi-variable functions). These will be populated accordingly depending on the transforms specified. If there are no transforms, then `transformed_value` will be equivalent to `value`, and `jacobian` will be zero. The attribute `transformed_value` will be used throughout inference since it is in the unconstrained space required by the algorithm. See [`World` and `Variable` API](../custom_proposers/variable.md) for more details.

## Specifying Transforms
Each proposer and inference method has the following optional parameters for initialization
```py
transform_type: TransformType
transforms: Optional[List]
```
TODO: Why is `transforms` a list? Can we have more than one transform associated to a variable? How does that work?

There are three TransformTypes which can be specified
* `TransformType.NONE`: no transform will be applied
* `TransformType.DEFAULT`: transforms will convert the distribution to the unconstrained space
* `TransformType.CUSTOM`: user-provided transforms will be applied

### Default Transforms

The transform applied to each variable depends on the constraints of its distribution:

* No constraints: no transform
* Lower bound $a$: $f(X) = \log(X - a)$
* Upper bound $b$: $f(X) = \log(-(X - b))$
* Simplex ($[0,1]$ interval): $f(X) = \text{stick breaking}(X)$
* Lower bound $a$ and upper bound $b$: $f(X) = \text{stick breaking}((X - a) / (b - a))$

TODO: how are these transforms represented in code?

TODO: not every user will understand what "stick breaking" has to do with lower and upper bounds. Can we provide a version with it with a name directly related the $[0,1]$ interval?

### Custom Transforms
If `TransformType.CUSTOM` is specified, the user must also provide a list of transforms to the `transforms` parameter of initialization.
```py
mh = SingleSiteNewtonianMonteCarlo(transform_type=TransformType.CUSTOM, transforms=[AffineTransform(2.0, 1.0)])
```

TODO: why are transforms attached to the inference algorithm? Shouldn't they be attached to a model, or a variable? What does it mean to provide a list of transforms to the algorithm? Does that mean the transform applies to all variable in the model? What happens if multiple transforms are provided?

For each transform, the user must provide the transform function, the inverse function, as well as the Jacobian calculation as described below. These transforms will be applied in order. For example, the list of transforms $[f, g]$ applied to $x$ will result in the value $g(f(x))$. It is recommended to implement the `Transform` class from PyTorch.
```py
def __call__(self, x):
  """
  Computes the forward transformation
  """
def inv(self, y):
  """
  Computes the inverse transformation
  """
def log_abs_det_jacobian(self, x, y):
  """
  Computes the log of the absolute value of determinant of the Jacobian `log |dy/dx|`
  """
```
