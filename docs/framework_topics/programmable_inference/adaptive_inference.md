# Adaptation and Warmup

MCMC inference methods all make use of some proposal distribution which should, through some justification, produce samples which resemble samples from the target distribution. Formally, the proposal distribution is $q_{\theta}(x,y)$ for $x,y \in \omega$.

Oftentimes, this proposal has some parameters $\theta$ which can best be chosen after inference has started, especially by using the observed data. An effective general MCMC method is to tune the proposal by spending a short time adapting the proposal distribution at the beginning of inference, before collecting proper samples from the posterior. While adaptation is occurring, the detailed balance equations are being violated, so the collected samples are not useful for posterior inference. However, this computation is typically worthwhile, as the adapted proposal distribution can be much more effective for collection healthy samples.

Bean Machine offers several adaptive inference methods, such as Newtonian Monte Carlo, Hamiltonian Monte Carlo, and Random Walk Metropolis Hastings. Here we describe the relatively standard API for using adaptive inference in each of these methods in Bean Machine.


## Adaptive API in Bean Machine

The single-site inference API has an argument `num_adapt_steps` in the call signature of `mf.infer()`. This number specifies how many steps from the beginning of the chain are used for adaptation. Accordingly, the argument `num_samples` specifies the number of post-adaptation inference steps. Using random walk as an example, this API is illustrated as follows.

```py
monte_carlo_samples = SingleSiteRandomWalk(
  step_size = 2.0,
).infer(
  queries,
  observations,
  num_samples,
  num_chains,
  num_adaptive_samples)
```

Adaptation is also accounted for by the API of the `MonteCarloSamples` class. Although the samples drawn during adaptation are not discarded, they are hidden by default. If desired, the samples drawn during adaptation can be accessed as shown below, using the argument `include_adapt_steps`. Furthermore, the samples drawn during adaptation are not used for computing diagnostics through the `Diagnostics` class.

```py
chain_0 = mcs.get_chain(0)
# This will actually give samples 0-100.
samples = chain_0.get_variable(queries[0], include_adapt_steps=True)[:100]
```

Under the hood, adaptive inference is done by calling `do_adaptation()` from the inference proposer class after each sample is drawn, for as many steps are as specified. Check out the API at `beanmachine/ppl/inference/proposer/AbstractSingleSiteProposer` to define this for your custom inference method!
