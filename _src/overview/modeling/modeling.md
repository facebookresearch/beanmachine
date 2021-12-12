---
title: 'Modeling'
sidebar_label: 'Modeling'
slug: '/overview/modeling'
---

## <a name="declarative_style"></a>Declarative Style

Bean Machine allows you to express models declaratively, in a way that closely follows the notation that statisticians use in their everyday work. Consider our example from the [Quick Start](../quick_start/quick_start.mdx). We could express this mathematically as:

* $n_\text{init}$: known constant
* $\texttt{reproduction\_rate} \sim \text{Exponential}(10.0)$
* $n_\text{new} \sim \text{Poisson}(\texttt{reproduction\_rate} \cdot n_\text{init})$

Let's take a look at the model again:

```py
reproduction_rate_rate = 10.0
num_init = 1087980

@bm.random_variable
def reproduction_rate():
    return dist.Exponential(rate=reproduction_rate_rate)

@bm.random_variable
def num_new(num_current):
    return dist.Poisson(reproduction_rate() *  num_current)
```

You can see how the Python code maps almost one-to-one to the mathematical definition. When building models in Bean Machine's declarative syntax, we encourage you to first think of the model mathematically, and then to evolve the code to fit to that definition.

Importantly, note that there is no formal class delineating your model. This means you're maximally free to build models that feel organic with the rest of your codebase and compose seamlessly with models found elsewhere in your codebase. Of course, you're also free to consolidate related modeling functionality within a class, which can help keep your model appropriately scoped!

## Random Variable Functions

Python functions annotated with `@bm.random_variable`, or _random variable functions_ for short, are the building blocks of models in Bean Machine. This decorator denotes functions which should be treated by the framework as random variables to learn about.

A random variable function must return a [PyTorch distribution](https://pytorch.org/docs/stable/distributions.html?highlight=distribution#module-torch.distributions) representing the probability distribution for that random variable, conditional on sample values for any other random variable functions that it depends on. For the most part, random variable functions can contain arbitrary Python code to model your problem! However, please do not depend on mutable external state (such as Python's `random` module), since Bean Machine will not be aware of it and your inference results may be invalid.

As outlined in the next two sections, calling random variable functions has different behaviors depending upon the callee's context.

## <a name="calling_inside"></a>Calling a Random Variable from Another Random Variable Function

When calling a random variable function from within another random variable function, you should treat the return value as a _sample_ from its underlying distribution. Bean Machine intercepts these calls, and will perform inference-specific operations in order to draw a sample from the underlying distribution that is consistent with the available observation data. Working with samples therefore decouples your model definition from the mechanics of inference going on under the hood.

**Calls to random variable functions are effectively memoized during a particular inference iteration.** This is a common pitfall, so it bears repeating: calls to the same random variable function with the same arguments will receive the same sampled value within one iteration of inference. This makes it easy for multiple components of your model to refer to the same logical random variable. This means that the common statistical notation discussed previously in [Declarative Style](#declarative_style) can easily map to your code: a programmatic definition like `reproduction_rate()` will always map to its corresponding singular statistical concept of $n_\text{new}$, no matter how many times it is invoked within a single model. This can also be appreciated from a _consistency_ point of view: if we define a new random variable `tautology` to be equal to `reproduction_rate() <= 3.0 or reproduction_rate() > 3.0`, the probability of `tautology` being `True` should be $1$, but if each invocation of `reproduction_rate` produced a different value, this would not hold. In [Defining Random Variable Families](#random_variable_families), we'll see how to control this memoization behavior with function parameters.

## <a name="calling_outside"></a>Calling a Random Variable from an Ordinary Function

It is valid to call random variable functions from ordinary Python functions. In fact, you've seen it a few times in the [Quick Start](../quick_start/quick_start.mdx) already! We've used it to bind data, specify our queries, and access samples once inference has been completed. Under the hood, Bean Machine transforms random variable functions so that they act like function references. Here's an example, which we just call from the Python toplevel scope:

```py
num_new()
```
```
RVIdentifier(function=<function num_new at 0x7ff00372d290>, arguments=())
```

As you can see, the call to this random variable function didn't return a distribution, or a sample from a distribution. Rather, it resulted in an `RVIdentifier` object, which represents a reference to a random variable function. You as the user can't do much with this object on its own, but Bean Machine will use this reference to access and re-evaluate different parts of your model.

## <a name="random_variable_families"></a>Defining Random Variable Families

As discussed in [Calling a Random Variable from Another Random Variable Function](#calling_inside), calls to a random variable function are memoized during a particular iteration of inference. How, then, can we create models with many random variables which have related but distinct distributions?

Let's dive into this by extending our model. In the previous example, we were modeling the number of new cases on a given day as a function of the number of infected individuals on the previous day. What if we wanted to model the spread of disease over multiple days? This might correspond to the following mathematical model:

* $n_i-n_{i-1} \sim \text{Poisson}(\texttt{reproduction\_rate} \cdot n_{i-1})$,
* where $n_i$ represents the number of cases on day $i$, and $n_0=n_\text{init}$.

It is common for statistical models to group random variables together into a random variable _family_ as you see here. In Bean Machine, the ability of _indexing_ into random variable families is generalized to arbitrary serializable Python objects. As an example, we could use a discrete time domain, here represented as a list of `datetime.date` objects,

```py
from datetime import date, timedelta

time = [date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3)]
```

in order to re-index the random varialble `num_new()` in our previous model:

```py
@bm.random_variable
def num_new(today):
    yesterday = today - timedelta(days=1)
    return dist.Poisson(reproduction_rate() * num_total(yesterday))
```

Note how this allows us to express a more complex dependency structure: where we previously relied on the argument `num_current` to describe the infections at some unspecified "current time", we can now use a more precise notion of (for example) "the day before `today`". This knowledge is in turn represented in another part of our probabilistic generative model, namely in the function `num_total`:

```py
# WARNING: INCORRECT COUNTER-EXAMPLE
def num_total(today):
    if today <= time[0]:
        return num_init
    else:
        yesterday = today - timedelta(days=1)
        return num_new(today) + num_total(yesterday)
```

## Transforming Random Variables

The problem in the above code is that we _can't_ decorate `num_total()` with `@bm.random_variable`. The reason we cannot is that it _doesn't return_ a [PyTorch elementary probability distribution](https://pytorch.org/docs/stable/distributions.html?highlight=distribution#module-torch.distributions). But, without a `@bm.random_variable` decorator on this function, Bean Machine _won't know_ that it should treat `num_new()` inside its body as a random variable function. As we discussed in [Calling a Random Variable from an Ordinary Function](#calling_outside), this call to `num_new()` would merely return an `RVIdentifier`, which is not what we want.

What do we do then? What we need here, and what is also the last important construct in Bean Machine's modeling toolkit, is the `@bm.functional` decorator. This decorator behaves like `@bm.random_variable`, except that it does require the function it is decorating to return only elementary distributions. As such, it can be used to deterministically transform the results of one or more other `@bm.random_variable` or `@bm.functional` functions. With this construct we can now write this model as follows:

```py
@bm.functional
def num_total(today):
    if today <= time[0]:
        return num_init
    else:
        yesterday = today - timedelta(days=1)
        return num_new(today) + num_total(yesterday)

@bm.random_variable
def num_new(today):
    yesterday = today - timedelta(days=1)
    return dist.Poisson(reproduction_rate() * num_total(yesterday))
```

One last note: while a `@bm.functional` can be queried during inference, it can't have observations bound to it.

---

Next, we'll look at how you can use [Inference](../inference/inference.md) to fit data to your model.
