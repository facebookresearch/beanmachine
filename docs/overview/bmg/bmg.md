---
id: bmg
title: The Bean Machine Graph (BMG) C++ Runtime
sidebar_label: 'C++ Runtime'
---

<!-- @import "../../header.md" -->

Bean Machine Graph (BMG) is an experimental C++ inference engine optimized for MCMC inference on static graphical models (i.e., models whose dependency structure is independent of all random variables’ values). Whereas Bean Machine models are defined through a series of random variables and functionals, Bean Machine Graph models are defined through the explicit construction of their computation graphs. Bean Machine Graph then performs inference directly on this graphical representation of the model.

## Modeling

Bean Machine Graph currently supports models with static graphs, as the full computation graph for the model must be explicitly defined at compile-time. The graphical representation of a model consists of a collection of nodes representing distributions, samples, constants, operators, and so on. For example, consider the following model:
```
@random_variable
def a():
    return Normal(0, 1.)

@random_variable
def b():
    return Normal(a() + 5, 1.)
```
In Bean Machine Graph, we represent it with the following graph:
![Typical DOT rendering of graph for model above](image.png)

Though Bean Machine Graph supports many commonly-used distributions and operators, it is less expressive than Bean Machine Python, which supports dynamic models, rich programmable inference, and so on.

## Inference

Bean Machine Graph currently supports Newtonian Monte Carlo, Gibbs sampling for Boolean variables, as well as rejection sampling. It performs inference efficiently by taking advantage of the static nature of its models. It also uses its own MCMC-focused automatic differentiation (AD) engine and performs the necessary gradient computations faster than other general-purpose AD engines which often have additional overhead.

## Using Bean Machine Graph

Due to the complexity of constructing the computation graph directly, users interested in Bean Machine Graph should use the [Beanstalk](../beanstalk/beanstalk) compiler, which enables users to write models using Bean Machine Python syntax and call the Bean Machine Graph inference engine as if it were another Bean Machine inference method.
To learn more about the current state of inference in Bean Machine Graph and experiment with different models, see the [Beanstalk documentation](../beanstalk/beanstalk).
