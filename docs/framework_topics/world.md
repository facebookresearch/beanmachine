---
id: world
title: 'Worlds and Variables'
sidebar_label: 'Worlds and Variables'
slug: '/world'
---

## Worlds
A [`World`](https://beanmachine.org/api/beanmachine.ppl.world.html) is Bean Machine's internal representation of the state of the model.  It can be thought of as a graph corresponding to a particular instantiation of the graphical model, or mathematically, joint distribution of the model $$p(data, random\_variables)$$.
When we run inference, we create worlds, each of which corresponds to a Monte Carlo sample of the posterior. New worlds are either accepted or rejected, which is (usually) determined by the accept-reject stage of
the MH algorithm.

The `World` class provides a flexible interface for performing inference and representing the intermediate or final results of inference. To that end, it provides a few functionalities.
Since a world is a representation of the model joint, it tracks which variables are latent and which are observed, and we can evaluate its density with its `log_prob` method which returns the
joint log probability given its instantiated variables.
A world can also be run as a [Python context manager](https://docs.python.org/3/library/contextlib.html) which allows the user to execute a model given a particular instantiation of a world.
Ordinarily, a `random_variable` returns a function pointer to the variable, but under the world context, the actual variable is sampled since we are instantiating it inside a world:

```py
@bm.random_variable
def foo():
    return Bernoulli(0.5)

pointer = foo()
assert isinstance(pointer, RVIdentifier)

world = World()
# everything run inside the world context manager
# is recorded in the world
with world:
    x = foo()

x == torch.tensor(1.)
x_var = world.get_variable(foo())
x_var.value == x
```
Since worlds are independent instantiations of the model, you can compose them interchangeably.  This allows us to inspect and manipulate our model as we see fit.
During MCMC inference, Bean Machine is constantly proposing new worlds in accordance with the proposal distribution, the collection of which form the posterior.

## Variables
`Variables` are primitives that contain metadata about a given random variable defined by `@bm.random_variable`, such as the distribution it was sampled from, its parents and children, the sampled value of the variable, and its log density.  They can represent latent or observed variables.  Only latent variables are inferred during inference and the values of the Variables can change between inference iterations.

## RVIdentifiers
Each [random variable](https://beanmachine.org/docs/overview/modeling/#random-variable-functions) is associated with a unique key `RVidentifier`. This is a pointer to the random variable and is implemented as a dataclass containing the random variable's Python function and arguments.
Since the function argument is a component of generating an `RVIdentifier`, the same callable can generate independent random variables by using different arguments:

```py
@bm.random_variable
def foo(i):
  return Normal(0., 1.)

foo(0)  # this is one variable with an RVIdentifier
foo(1)  # this is another variable with a different RVIdentifier
```
