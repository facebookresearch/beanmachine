---
id: analysis
title: Analysis
sidebar_label: 'Analysis'
---

<!-- @import "../../header.md" -->

Inference results are useful not only for learning from your posterior distributions, but for verifying that inference ran correctly. We'll cover common techniques for analyzing results in this section.

## Results of Inference

Bean Machine stores the results of inference in an object of type `MonteCarloSamples`. This class uses
an underlying data structure of a dictionary mapping a model's random variables to a PyTorch tensor of sampled values. The class can be accessed like a dictionary, and there are additional wrapper methods to make function calls more explicit.

In the [Inference](../inference/inference.md) section, we obtained results on the disease modeling example by running inference:

```py
samples = bm.CompositionalInference().infer(
    queries=[ reproduction_rate() ],
    observations=observations,
    num_samples=10000,
    num_chains=4,
)
samples
```
```
<beanmachine.ppl.inference.monte_carlo_samples.MonteCarloSamples>
```

### Extracting Samples for a Specific Variable

We ran inference to compute the posterior for `reproduction_rate()`, since that was listed in `queries`. We can see that the posterior for `reproduction_rate()` (and only for `reproduction_rate()`) is available in `samples`:

```py
samples.get_rv_names()
```
```
[RVIdentifier(function=<function reproduction_rate>, arguments=())]
```

To extract the inference results for `reproduction_rate()`, we can use `get_variable()`:

```py
samples.get_variable(reproduction_rate())
```
```
tensor([[1.0000, 0.4386, 0.2751,  ..., 0.2177, 0.2177, 0.2193],
        [0.2183, 0.2183, 0.2184,  ..., 0.2177, 0.2177, 0.2177],
        [0.2170, 0.2180, 0.2183,  ..., 0.2180, 0.2180, 0.2180],
        [0.2180, 0.2180, 0.2172,  ..., 0.2180, 0.2180, 0.2176]])
```

The result has shape $4 \times 10000$, representing the 10,000 samples that we drew in each of the four chains of inference from the posterior distribution.

`MonteCarloSamples` supports convenient dictionary indexing syntax to obtain the same information:

```py
samples[ reproduction_rate() ]
```
```
tensor([[1.0000, 0.4386, 0.2751,  ..., 0.2177, 0.2177, 0.2193],
        [0.2183, 0.2183, 0.2184,  ..., 0.2177, 0.2177, 0.2177],
        [0.2170, 0.2180, 0.2183,  ..., 0.2180, 0.2180, 0.2180],
        [0.2180, 0.2180, 0.2172,  ..., 0.2180, 0.2180, 0.2176]])
```

Please note that many inference methods require a small number of samples before they start drawing samples that correctly resemble the posterior distribution. We recommend you discard at least a few hundred samples before using your inference results.

### Extracting Samples for a Specific Chain

We'll see how to make use of chains in [Diagnostics](#diagnostics); for inspecting the samples themselves, it is often useful to examine each chain individually. The recommended way to access the results of a specific chain is with `get_chain()`:

```py
chain = samples.get_chain(chain=0)
chain
```
```
<beanmachine.ppl.inference.monte_carlo_samples.MonteCarloSamples>
```

This returns a new `MonteCarloSamples` object which is limited to the specified chain. Tensors no longer have a dimension representing the chain:

```py
chain[ reproduction_rate() ]
```
```
tensor([1.0000, 0.4386, 0.2751,  ..., 0.2177, 0.2177, 0.2193])
```

### Visualizing Distributions

Visualizing the results of inference can be a great help in understanding them. Since you now know how to access posterior samples, you're free to use whatever visualization tools you prefer.

```py
reproduction_rate_samples = samples[ reproduction_rate() ][0][100:]
plt.hist(reproduction_rate_samples, label="reproduction_rate_samples")
plt.axvline(reproduction_rate_samples.mean(), label=f"Posterior mean = {reproduction_rate_samples.mean() :.2f}", color="K")
plt.xlabel("reproduction_rate")
plt.ylabel("Probability density")
plt.legend();
```

![](/img/posterior_reproduction_rate.png)

## <a name="diagnostics"></a>Diagnostics

After running inference it is useful to run diagnostic tools to assess reliability of the inference run. Bean Machine provides two standard types of such diagnostic tools, discussed below.

### Summary Statistics

Bean Machine provides important summary statistics for individual, numerically-valued random variables. Let's take a look at the code to generate them, and then we'll break down the statistics themselves.

Bean Machine's `Diagnostics` object makes it easy to generate a Pandas `DataFrame` presenting these statistics for all queried random variables:

```py
bm.Diagnostics(samples).summary()
```

| | avg | std | 2.5% | 50% | 97.5% | r_hat | n_eff
| -- | -- | -- | -- | -- | -- | -- | --
| `reproduction_rate()[]` | 0.218 | 0.004 | 0.216 | 0.218 | 0.219 | 1.003 | 631.315

The statistics presented are:

  1. **Mean and standard deviation.**
  2. **95% confidence interval.**
  3. **Convergence diagnostic [$\hat{R}$](https://projecteuclid.org/euclid.ss/1177011136).**
  4. **Effective sample size [$N_\text{eff}$](https://www.mcmchandbook.net/HandbookChapter1.pdf).**

These statistics provide different insights into the quality of the results of inference. For instance, we can use them in combination with synthetically generated data for which we know ground truth values for parameters and then check to make sure that these values fall within the 95% confidence intervals. Of course, in doing so it is important to keep in mind that the prior distributions in our model (and not just the data) will always have an influence on the posterior distribution. Similarly, we can use the size of the 95% confidence interval to gain insights into the model: If it is large, this could indicate that either we have too few observations or that the prior is too weak.

$\hat{R} \in [1, \infty)$ summarizes how effective inference was at converging on the correct posterior distribution for a particular random variable. It uses information from all chains run in order to assess whether inference had a good understanding of the distribution or not. Values very close to $1.0$ indicate that all chains discovered similar distributions for a particular random variable. We do not recommend using inference results where $\hat{R} > 1.1$, as inference may not have converged. In that case, you may want to run inference for more samples. However, there are situations in which increasing the number of samples will not improve convergence. In this case, it is possible that the prior is too far from the posterior that inference is unable to reliably explore the posterior distribution.

$N_\text{eff} \in [1,$ `num_samples`$]$ summarizes how independent posterior samples are from one another. Although inference was run for `num_samples` iterations, it's possible that those samples were very similar to each other (due to the way inference is implemented), and may not each be representative of the full posterior space. Larger numbers are better here, and if your particular use case calls for a certain number of samples to be considered, you should ensure that $N_\text{eff}$ is at least that large.

In the case of our example model, we have a healthy $\hat{R}$ value close to 1.0, and a healthy number of effective samples of 631.

### Diagnostic Plots

Bean Machine can also plot diagnostic information to assess health of the inference run. Let's take a look:

```py
bm.Diagnostics(samples).plot(display=True)
```

![](/img/trace_reproduction_rate.png)
![](/img/autocorrelation_reproduction_rate.png)

The diagnostics output shows two diagnostic plots for individual random variables: trace plots and autocorrelation plots.

  * Trace plots are simply a time series of values assigned to random variables over each iteration of inference. The concrete values assigned are usually problem-specific. However, it's important that these values are "mixing" well over time. This means that they don't tend to get stuck in one region for large periods of time, and that each of the chains ends up exploring the same space as the other chains throughout the course of inference.
  * Autocorrelation plots measure how predictive the last several samples are of the current sample. Autocorrelation may vary between -1.0 (deterministically anticorrelated) and 1.0 (deterministically correlated). (We compute autocorrelation approximately, so it may sometimes exceed these bounds.) In an ideal world, the current sample is chosen independently of the previous samples: an autocorrelation of zero. This is not possible in practice, due to stochastic noise and the mechanics of how inference works. The autocorrelation plots here plot how correlated samples from the end of the chain are compared with samples taken from elsewhere within the chain, as indicated by the iteration index on the x axis.

For our example model, we see from the trace plots that each of the chains are healthy: they don't get stuck, and do not explore a chain-specific subset of the space. From the autocorrelation plots, we see the absolute magnitude of autocorrelation to be very small, often around 0.1, indicating a healthy exploration of the space.

---

Congratulations, you've made it through the **Overview**! If you're looking to get an even deeper understanding of Bean Machine, check out the **Framework** topics next. Or, if you're looking to get to coding, check out our [Tutorials](../tutorials/tutorials.md). In either case, happy modeling!
