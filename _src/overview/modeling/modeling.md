---
id: modeling
title: 'Modeling'
sidebar_label: 'Modeling'
---

<!-- @import "../../header.md" -->

## <a name="declarative_style"></a>Declarative Style

Bean Machine allows you to express models declaratively, in a way that closely follows the notation that statisticians use in their everyday work. Consider our example from the [Quick Start](../quick_start/quick_start.md). We could express this mathematically as:

* $n_\text{infected}$: known constant
* $\texttt{reproduction\_rate} \sim \text{Exponential}(10.0)$
* $n_\text{new} \sim \text{Poisson}(\texttt{reproduction\_rate} \cdot n_\text{infected})$

Let's take a look at the model again:

```py
num_infected = 1087980

@bm.random_variable
def reproduction_rate():
    return dist.Exponential(rate=10.0)

@bm.random_variable
def num_new_cases():
    return dist.Poisson(reproduction_rate() *  num_infected)
```

You can see how the Python code maps almost one-to-one to the mathematical definition. When building models in Bean Machine's declarative syntax, we encourage you to first think of the model mathematically, and then to evolve the code to fit to that definition.

Importantly, note that there is no formal class delineating your model. This means you're maximally free to build models that feel organic with the rest of your codebase and compose seamlessly with models found elsewhere in your codebase. Of course, you're also free to consolidate related modeling functionality within a class, which can help keep your model appropriately scoped!

## Random Variable Functions

Python functions annotated with `@bm.random_variable`, or "random variable functions" for short, are the building blocks of models in Bean Machine. This decorator denotes functions which should be treated by the framework as random variables to learn about.

A random variable function must return a [PyTorch distribution](https://pytorch.org/docs/stable/distributions.html?highlight=distribution#module-torch.distributions) representing the probability distribution for that random variable, conditional on sample values for any other random variable functions that it depends on. For the most part, random variable functions can contain arbitrary Python code to model your problem! However, please do not depend on mutable external state (such as Python's `random` module), or your inference results may be invalid, since Bean Machine will not be aware of it.

Calling random variable functions has different behaviors depending upon the callee's context, outlined in the next two sections.

## <a name="calling_inside"></a>Calling a Random Variable from Another Random Variable Function

When calling a random variable function from within another random variable function, you should treat the return value as a _sample_ from its underlying distribution. Bean Machine intercepts these calls, and will perform inference-specific operations in order to draw a sample from the underlying distribution that is consistent with the available observation data. Working with samples therefore decouples your model definition from the mechanics of inference going on under the hood.

**Calls to random variable functions are effectively memoized during a particular inference iteration.** This is a common pitfall, so it bears repeating: calls to the same random variable function with the same arguments will receive the same sampled value within one iteration of inference. This makes it easy for multiple components of your model to refer to the same logical random variable. This means that the common statistical notation discussed previously in [Declarative Style](#declarative_style) can easily map to your code: a programmatic definition like `reproduction_rate()` will always map to its corresponding singular statistical concept of $n_\text{new}$, no matter how many times it is invoked within a single model. This can also be appreciated from a _consistency_ point of view: if we define a new random variable `tautology` to be equal to `reproduction_rate() <= 3.0 or reproduction_rate() > 3.0`, the probability of `tautology` being `True` should be $1$, but if each invocation of `reproduction_rate` produced a different value, this would not hold.

In [Defining Random Variable Families](#random_variable_families), we'll see how to control this memoization behavior with function parameters.

## <a name="calling_outside"></a>Calling a Random Variable from an Ordinary Function

It is valid to call random variable functions from ordinary Python functions. In fact, you've seen it a few times in the [Quick Start](../quick_start/quick_start.md) already! We've used it to bind data, specify our queries, and access samples once inference has been completed.

Under the hood, Bean Machine transforms random variable functions so that they act like function references. Here's an example, which we just call from the Python toplevel scope:

```py
num_new_cases()
```
```
RVIdentifier(function=<function num_new_cases at 0x7ff00372d290>, arguments=())
```

As you can see, the call to this random variable function didn't return a distribution, or a sample from a distribution. Rather, it resulted in an `RVIdentifier` object, which represents a reference to a random variable function. You as the user can't do much with this object on its own, but Bean Machine will use this reference to access and re-evaluate different parts of your model.

## <a name="random_variable_families"></a>Defining Random Variable Families

As discussed in [Calling a Random Variable from Another Random Variable Function](#calling_inside), calls to a random variable function are memoized during a particular iteration of inference. How, then, can we create models with many random variables which have related but distinct distributions?

Let's dive into this by extending our example model. In the previous example, we were modeling the number of new cases on a given day as a function of the number of infected individuals on the previous day. However, what if we wanted to  model the spread of disease over multiple days? This might correspond to the following mathematical model:

* $n_i \sim \text{Poisson}((1 + \texttt{reproduction\_rate}) \cdot n_{i-1})$, where $n_i$ represents the number of cases on day $i$.

It is common for statistical models to group random variables together into a _family_ of random variables as you see here.

In Bean Machine, we generalize the ability to index into a family of random variables with arbitrary Python objects. We can extend our previous example to add an index onto our random variable `num_new_cases()` with an object of type `datetime.date`:

```py
import datetime

@bm.random_variable
def num_cases(day):
    # Base case for recursion
    if day == datetime.date(2021, 1, 1):
        return dist.Poisson(num_infected)
    return dist.Poisson(
        (1 + reproduction_rate()) *  num_cases(day - datetime.timedelta(days=1))
    )
```

## Transforming Random Variables

There's one last important construct in Bean Machine's modeling toolkit: `@bm.functional`. This decorator is used to deterministically transform the results of one or more random variables.

In the above example, you'll notice that we added 1 to the reproduction rate, to turn it into a coefficient for the previous day's number of cases. It would be nice to capture this as its own function. Here's an **incorrect** attempt (don't do this!):

```py
# COUNTER-EXAMPLE

def infection_rate():
    return 1 + reproduction_rate()

@bm.random_variable
def num_cases(day):
    # Base case for recursion
    if day == datetime.date(2021, 1, 1):
        return dist.Poisson(num_infected)
    return dist.Poisson(
        infection_rate() *  num_cases(day - datetime.timedelta(days=1))
    )
```

Why is this incorrect? You'll notice that `num_cases()` now calls into `infection_rate()`, which itself depends on the random variable function `reproduction_rate()`. We _can't_ make `infection_rate()` a random variable function, as it does _not_ return a [PyTorch distribution](https://pytorch.org/docs/stable/distributions.html?highlight=distribution#module-torch.distributions). However, since there is no `@bm.random_variable` decorator, Bean Machine inference _won't know_ that it should treat `reproduction_rate()` inside the function scope as a random variable function. Indeed, like we discussed in [Calling a Random Variable from an Ordinary Function](#calling_outside), `reproduction_rate()` in this context would merely return an `RVIdentifier` -- definitely not what we want.

What do we do then? Bean Machine's `@bm.functional` decorator is here to serve this exact purpose! `@bm.functional` behaves like `@bm.random_variable` except that it does not return a distribution. As such, it is appropriate to use to deterministically transform the results of one or more other `@bm.random_variable` or `@bm.functional` functions.

Here's the correct way to write this model:

```py
@bm.functional
def infection_rate():
    return 1 + reproduction_rate()

@bm.random_variable
def num_cases(day):
    # Base case for recursion
    if day == datetime.date(2021, 1, 1):
        return dist.Poisson(num_infected)
    return dist.Poisson(
        infection_rate() *  num_cases(day - datetime.timedelta(days=1))
    )
```

One last note: while a `@bm.functional` can be queried (viewed) during inference, it can't be directly bound (softly constrained) to observations like a `@bm.random_variable`. This is because it is a deterministic function and thus inappropriate as a likelihood.

---

Next, we'll look at how you can use [Inference](../inference/inference.md) to fit data to your model.
