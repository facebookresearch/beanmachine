---
title: Inference
sidebar_label: 'Inference'
slug: '/overview/inference'
---

Inference is the process of combining a _model_ with _data_ to obtain _insights_, in the form of probability distributions over values of interest.

A little note on vocabulary: You've already seen in [Modeling](../modeling/modeling.md) that the _model_ in Bean Machine is comprised of random variable functions. In Bean Machine, the _data_ is built up of a dictionary mapping random variable functions to their observed values, and _insights_ take the form of discrete samples from a probability distribution. We refer to the random variables for which we're learning distributions as _queried random variables_.

Let's make this concrete by returning to our disease modeling example. As a refresher, here's the full model:

```py
reproduction_rate_rate = 10.0
num_init = 1087980
time = [date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3)]

@bm.random_variable
def reproduction_rate():
    return dist.Exponential(rate=reproduction_rate_rate)

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

## Prior and Posterior Distributions

The $\text{Exponential}$ distribution used here represents our beliefs about the disease's reproduction rate before seeing any data, and is known as the _prior distribution_. We've visualized this distribution previously: it represents a reproduction rate that is around 10% on average, but could be as high as 50%, and is highly right-skewed (the right side has a long tail). Values associated with prior distributions (here `reproduction_rate()`) are known as _latent variables_.

While the prior distribution encodes our prior beliefs, inference will perform the important task of adjusting latent variable values so that they balance both our prior belief and our knowledge from observed data. We refer to this distribution, after conditioning on observed data, as a _posterior distribution_. And the remaining parts of the generative model, which determine the notion of consistency used to match the latent variables with the observations, are collectively called the _likelihood terms_ of the model (here consisting of `num_total(today)` and `num_new(today)`). The way inference is performed depends upon the specific numerical method used, but it does always mean that inferred distributions will blend smoothly from resembling your prior distribution, when there is little data observed, to more wholly representing your observed data, when there are many observations.

## <a name="binding-data"></a>Binding Data

Inference requires us to bind data to the model in order to learn posterior distributions for our queried random variables. This is achieved by passing an `observations` dictionary to Bean Machine at inference time. Instead of sampling from random variables contained in that dictionary, Bean Machine will consider them to take on the constant values provided, and will try to find values for other random variables in your model that are consistent with the `observations`. For this example model, we can bind a few days of data as follows, taking care to match the $\text{Poisson}$ distributions in `num_new()` with the corresponding _increases_ in infection counts which they're modelling:

```py
case_history = tensor([num_init, 1381734., 1630446.])
observations = {num_new(t): d for t, d in zip(time[1:], case_history.diff())}
```

Though correct, that code is a bit difficult to read for pedagogical purposes. The following code is equivalent:

```py
observations = {
    num_new(date(2021, 1, 2)): tensor(293754.),
    num_new(date(2021, 1, 3)): tensor(248712.),
}
```

Recall that calls to random variable functions from ordinary functions (including the Python toplevel) return `RVIdentifier` objects. So, the keys of this dictionary are `RVIdentifiers`, and the values are values of observed data corresponding to each key that you provide. Note that the value for a particular observation must be of the same type as the [support for the distribution that it's bound to](https://pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution.support). In this case, the [support for the $\text{Poisson}$ distribution](https://pytorch.org/docs/stable/distributions.html#torch.distributions.poisson.Poisson.support) is scalar and non-negative, so what we have bound here are bounded scalar tensors.

## Running Inference

We're finally ready to run inference! Let's take a look first, and then we'll explain what's happening.

```py
samples = bm.CompositionalInference().infer(
    queries=[reproduction_rate()],
    observations=observations,
    num_samples=7000,
    num_adaptive_samples=3000,
    num_chains=4,
)
```

Let's break this down. There is an inference method (in this example, that's the `CompositionalInference` class), and there's a call to `infer()`.

Inference methods are simply classes that extend from `AbstractInference`. These classes define the engine that will be used in order to fit posterior distributions to queried random variables given observations. In this particular example, we've chosen to use the specific inference method `CompositionalInference` to run inference for our disease modeling problem.

`CompositionalInference` is a powerful, flexible class for configuring inference in a variety of ways. By default, `CompositionalInference` will select an inference method for each random variable that is appropriate based on its support. For example, for differentiable random variables, this inference method will attempt to leverage gradient information when generating samples from the posterior; for discrete random variables, it will use a uniform sampler to get representative draws for each discrete value.

A full discussion of the powerful `CompositionalInference` method, including extensive instructions on how to configure it to tailor specific inference methods for particular random variables, can be found in the [Compositional Inference](../../framework_topics/programmable_inference/compositional_inference.md) guide. Bean Machine offers a variety of other inference methods as well, which can perform differently based on the particular model you're working with. You can learn more about these inference methods under the [Inference](../../framework_topics/inference/inference.md) framework topic.

Regardless of the inference method, `infer()` has a few important general parameters:

| Name | Usage
| --- | ---
| `queries` | A list of random variable functions to fit posterior distributions for.
| `observations` | The Python dictionary of observations that we discussed in [Binding Data](#binding-data).
| `num_samples` | The integer number of samples with which to approximate the posterior distributions for the values listed in `queries`.
| `num_adaptive_samples` | The integer number of samples to spend before `num_samples` on tuning the inference algorithm for the `queries`, see [Adaptation and Warmup](../../framework_topics/programmable_inference/adaptive_inference.md).
| `num_chains` | The integer number of separate inference runs to use. Multiple chains can be used to verify that inference ran correctly.

You've already seen `queries` and `observations` many times. `num_adaptive_samples` and `num_samples` are used to specify the number of iterations to respectively tune, and then run, inference. More iterations will allow inference to explore the posterior distribution more completely, resulting in more reliable posterior distributions. `num_chains` lets you specify the number of identical runs of the entire inference algorithm to perform, called "chains". Multiple chains of inference can be used to validate that inference ran correctly and was run for enough iterations to produce reliable results, and their behavior can also help detect whether the model was well specified. We'll revisit chains in [Inference Methods](../../framework_topics/inference/inference.md).

---

Now that we've run inference, it's time to explore our results in the [Analysis](../analysis/analysis.mdx) section!
