---
id: quick_start
title: 'Quick Start'
sidebar_label: 'Quick Start'
---
import useBaseUrl from '@docusaurus/useBaseUrl';

<!-- @import "../../header.md" -->

Let's quickly translate the model we discussed in the [Introduction](../introduction/introduction.md) into Bean Machine code! Although this will get you up-and-running, **it's important that you read through all of the pages in the Overview to have a complete understanding of Bean Machine**. Happy modeling!

## Modeling

As a quick refresher, we're writing a model to understand a disease's reproduction rate, based on the number of new cases of that disease we've seen. Though we never observe the true reproduction rate, let's start off with a prior distribution that represents our beliefs about the reproduction rate before seeing any data.

```py
import beanmachine.ppl as bm
import torch.distributions as dist

@bm.random_variable
def reproduction_rate():
    # Exponential distribution with rate 10 has mean 0.1.
    return dist.Exponential(rate=10.0)
```

There are a few things to notice here!

  * Most importantly, we've decorated this function with `@bm.random_variable`. This is how you tell Bean Machine to interpret this function probabilistically. `@bm.random_variable` functions are the building blocks of Bean Machine models, and let the framework explore different values that the function represents when fitting a good distribution for observed data that you'll provide later.
  * Next, notice that the function returns a [PyTorch distribution](https://pytorch.org/docs/stable/distributions.html?highlight=distribution#module-torch.distributions). This distribution encodes your prior belief about a particular random variable. In the case of $\text{Exponential}(10.0)$, our prior has this shape:

<img src={useBaseUrl("/img/exponential_10.png")} />

    As you can see, the prior encourages smaller values for the reproduction rate, averaging at a rate of 10%, but allows for the possibility of much larger spread rates.
  * Lastly, realize that although you've provided a prior distribution here, the framework will automatically "refine" this distribution, as it searches for values that represent observed data that you'll provide later. So, after we fit the model to observed data, the random variable will no longer look like the graph shown above!

The last piece of the model is how the reproduction rate relates to the new cases of illness that we observe the subsequent day. This number of new cases is related to the underlying reproduction rate -- how fast the virus tends to spread -- as well as the current number of cases. However, it's not a deterministic function of those two values. Instead, it depends on a lot of environmental factors like social behavior, stochasticity of transmission, and so on. It would be far too complicated to capture all of those factors in a single model. Instead, we'll aggregate all of these environmental factors in the form of a probability distribution, the $\text{Poisson}$ distribution.

Let's say, for this example, we observed a little over a million, 1087980, cases today. We use such a precise number here to remind you that this is a known value and not a random one. In this case, if the disease were to happen to have a reproduction rate of 0.1, this is what our $\text{Poisson}$ distribution for new cases would look like:

<img src={useBaseUrl("/img/poisson.png")} />

Let's write this up in Bean Machine. Using the syntax we've already seen, it's pretty simple:

```py
num_infected = 1087980

@bm.random_variable
def num_new_cases():
    return dist.Poisson(reproduction_rate() *  num_infected)
```

As you can see, this function relies on the `reproduction_rate()` that we defined before. Do notice: even though `reproduction_rate()` returns a distribution, here the return value from `reproduction_rate()` is treated like a sample from that distribution! Bean Machine works hard behind the scenes to sample efficiently from distributions, so that you can easily build sophisticated models that only have to reason about these samples.

## Data

With the model fully defined, we should gather some data to learn about! In the real world, you might work with a government agency to determine the number of real, new cases observed on the next day. For sake of our example, let's say that we observed 238154 new cases on the next day. Bean Machine's random variable syntax allows you to bind this information directly as an observation for the `num_new_cases()` random variable within a simple Python dictionary. Here's how to do it:

```py
from torch import tensor

observations = {
    # PyTorch distributions expect tensors, so we provide a tensor here.
    num_new_cases(): tensor(238154.),
}
```

Using a random variable function as keys in this dictionary may feel unusual at first, but it quickly becomes an intuitive way to reference these random variable functions by name!

## Inference

With model and observations in hand, we're ready for the fun part: inference! Inference is the process of combining _model_ with _data_ to obtain _insights_, in the form of probability distributions over values of interest. Bean Machine offers a powerful and general inference framework to enable fitting arbitrary models to data.

The call to inference involves first creating an appropriate inference engine object and then invoking the `infer` method:

```py
samples = bm.CompositionalInference().infer(
    queries=[ reproduction_rate() ],
    observations=observations,
    num_samples=10000,
)
```

There's a lot going on here! First, let's take a look at the inference method that we used, `CompositionalInference()`. Bean Machine supports generic inference, which means that it can fit your model to the data without knowing the intricate and particular workings of the model that you defined. However, there are lots of ways of performing this, and Bean Machine supports a rich library of inference methods that can work for different kinds of models. For now, all you need to know is that `CompositionalInference` is a general inference strategy that will try to automatically determine the best inference method(s) to use for your model, based on the definitions of random variables you've provided. It should work well for this simple model. You can check out our guides on [Inference](../inference/inference.md) to learn more!

Let's take a look at the parameters to `infer()`. In `queries`, you provide a list of random variables that you're interested in learning about. Bean Machine will learn probability distributions for these, and will return them to you when inference completes! Note that this uses exactly the same pattern to reference random variables that we used when binding data.

We bind our real-world observations with the `observations` parameter. This provides a set of probabilistic constraints that Bean Machine seeks to satisfy during inference. In particular, Bean Machine tries to fit probability distributions for unobserved random variables, so that those probability distributions explain the observed data -- and your prior beliefs -- well.

Lastly, `num_samples` is the number of samples that you want to learn. Bean Machine doesn't learn smooth probability distributions for your `queries`, but instead accumulates a representative set of samples from those distributions. This parameter lets you specify how many samples should comprise these distributions.

## Analysis

Our results are ready! Let's visualize results for the reproduction rate.

The `samples` object that we have now contains probability distributions that we've fit for our model and data. It supports dictionary-like indexing using -- you guessed it -- the same random variable referencing syntax we've seen before.

<!-- TODO: The syntax for accessing samples is ugly because of chains and detaching. We should fix it. -->
```py
reproduction_rate_samples = samples[ reproduction_rate() ][0]
reproduction_rate_samples
```

```
tensor([0.0146, 0.1720, 0.1720,  ..., 0.2187, 0.2187, 0.2187])
```

Let's visualize that more intuitively.

```py
import matplotlib.pyplot as plt

plt.hist(reproduction_rate_samples, label="reproduction_rate_samples")
plt.axvline(reproduction_rate_samples.mean(), label=f"Posterior mean = {reproduction_rate_samples.mean() :.2f}", color="K")
plt.xlabel("reproduction_rate")
plt.ylabel("Probability density")
plt.legend();
```

<img src={useBaseUrl("/img/results.png")} />

This histogram represents our beliefs over the underlying reproduction rate, after observing the current day's worth of new cases. You'll note that it balancing our prior beliefs with rate that we learn just from looking at the new data. It also captures the uncertainty inherent in our estimate!

## We're not done yet!

This is the tip of the iceberg. The rest of this **Overview** will cover critical concepts from the above sections. Read on to learn how to make the most of Bean Machine's powerful modeling and inference systems!

<!-- ## Quick start Part 1

### Modeling

At this point it would be helpful to briefly look at our example model written in Bean Machine.
Random variables such as `reproduction_rate` and `num_new_cases` are represented as functions which compute the Conditional Probability Distribution (CPD) of these variables and return a [PyTorch distribution object](https://pytorch.org/docs/stable/distributions.html?highlight=distribution#module-torch.distributions).
These **dependency functions** that compute the CPD are marked with the `beanmachine.ppl.random_variable` decorator and all Elementary Probability Distributions (EPDs) are based on PyTorch EPDs in the `torch.distribution` package.

```py
import beanmachine.ppl as bm
import torch.distributions as dist
from torch import tensor


num_infected = 1087980

@bm.random_variable
def reproduction_rate():
    return dist.Exponential(1 / 0.1)

@bm.random_variable
def num_new_cases():
    return dist.Poisson(reproduction_rate() *  num_infected)
```

### Inference

The call to inference involves first creating an appropriate inference engine object and then invoking the `infer` method as follows

```py
engine = bm.CompositionalInference()
samples = engine.infer(
    queries = [reproduction_rate()],
    observations = {num_new_cases() : tensor(238154.0)},
    num_samples = 1000,
)
```

Here it is important to note that the call `reproduction_rate()` in the list of queried variables refers to the name of a random variable while the same call in the dependency function  of `num_new_cases()` refers to the value of the random variable.

The semantics of the `infer` call are to request the posterior distribution of the random variable `reproduction_rate()` given that the value of the random variable `num_new_cases()` is equal to `238154`.
Note that all values in Bean Machine are represented as PyTorch tensors.

### Results

The inference engine that we are using in this example is an instance of Markov Chain Monte Carlo (MCMC) inference which represents the posterior as a set of samples.
We can inspect these samples by using the function call `reproduction_rate()` to refer to the name of a random variable as follows,

```py
samples[reproduction_rate()]
```

```
tensor([[0.2189, 0.2187, 0.2188,  ..., 0.2186, 0.2187, 0.2188],
        [0.2187, 0.2192, 0.2185,  ..., 0.2187, 0.2182, 0.2183],
        [0.2181, 0.2197, 0.2188,  ..., 0.2185, 0.2191, 0.2187],
        [0.2188, 0.2184, 0.2183,  ..., 0.2183, 0.2189, 0.2189]],
       grad_fn=<SliceBackward>)
```

By default inference produces samples from 4 independent chains, hence the 4 arrays above.
This can be controlled with the `num_chains` parameter.
Of course, we can do the usual PyTorch tensor operations on these samples,

```py
samples[reproduction_rate()].mean().item()
```

```
0.2188912332057953
```

### Diagnostics

The Bean Machine library has standard MCMC diagnostics available.
These diagnostics can be accessed through `Diagnostics` as follows

```py
diag = bm.Diagnostics(samples)
diag.summary()
```

|                              |avg       |std      |2.5%       |50%     |97.5%     |r_hat        |n_eff|
|---|---|---|---|---|---|---|---|
|reproduction_rate()[]  |0.218891  |0.000454  |0.218011  |0.218889  |0.219761  |1.003263  |2356.885498|

In addition to the numerical diagnostics, other standard graphs can also be produced as follows

```py
diag.plot(display=True)
```

![Trace Plot](trace1.png)

![Auto Correlation Plot](autocorr1.png) -->
