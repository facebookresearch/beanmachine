---
title: 'Custom Proposers'
sidebar_label: 'Custom Proposers'
slug: '/custom_proposers'
---
## API

Bean Machine is flexible and allows users to supply custom MCMC proposers. This enables users to easily incorporate domain knowledge or exploit model structure during inference.

We will focus on [Metropolis Hastings algorithm](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm), the most common flavor of MCMC inference.
For each iteration of sampling of variable $X$ with value $x$ using the MH algorithm, we first sample a new value $x'$ for $X$ using proposal distribution $g$.
We then update the world to reflect this change, and use the Metropolis Hastings ratio of
$$\text{min}\big[1, \frac{p(x')g(x \mid x')}{p(x)g(x' \mid x)}\big]$$
to decide whether to keep the update or to revert to the world as it was before it.

Bean Machine allows users to easily provide custom proposals for $g$, without needing to implement other details such as the calculations for $p(x')$ or the acceptance ratio.
All MH proposers must inherit from the `BaseSingleSiteMHProposer` class and implement the following method, which returns the proposal distribution as a `torch.distribution`:
```py
def get_proposal_distribution(self, world: World) -> dist.Distribution
```
and optionally an adaptation stage if applicable:
```py
def do_adaptation(self, world: World, accept_log_prob: torch.Tensor, *args, **kwargs) -> None:
```

One may also override the `propose` method which is invoked at each iteration, to use any algorithm to propose a new world along with its acceptance log probability. By default the MH algorithm is used.
```py
def propose(self, world: World) -> Tuple[World, torch.Tensor]
```

For single site algorithms, passing the custom proposer into `SingleSiteInference` and using the resulting object, which assigns an instance of the proposer per variable, is usually sufficient.
To implement custom blocking, which is handled at the algorithm level, the user must define the abstract method `BaseInference.get_proposers` which defines
which proposers (and which order) to execute for which random variables.  See [blocking](../custom_inference/block_inference.md) for more examples.

That's it! Let's walk through an example to see how we'd do this in practice.

## Example Custom Proposer

<!--
It might make sense to replace this example by a more classic Metropolis-Hastings example).
It is a bit more complicated than needed, using a Laplace distribution that is not one of the standard ones.
It involves inverting this distribution but does not actually show the details of doing so.
Also, the proposal distribution doesn't depend on the current value of the variable, which is a non-typical situation for proposal distributions.
-->

Here, we implement a custom proposer for locating seismic events on Earth as described in [1]. In particular, the simplified version of the model is used, where one seismic event, denoted in the model by `event_attr()`, occurs randomly on the surface of the Earth. This event sends seismic energy in a single wave. There are different seismic stations across the surface of the Earth, and each station will noisily record its attributes of time, azimuth, and slowness, denoted by `det_attr(s)`. The inference problem is to find the `event_attr()` given the `det_attr(s)`.

```py
@bm.random_variable
def event_attr():
  return SeismicEventPrior()

@bm.random_variable
def det_attr(station):
    det_loc = calculate_detection(station, event_attr())
    return Laplace(det_loc, scale)
```

There is domain knowledge within seismology to mathematically solve for the most likely attributes of an event given the detection attributes for a specific station. Due to the noisy nature of the data, these predicted locations can be inaccurate, but this can be mitigated by considering the detections in many stations.

An ideal MH proposer for location values would propose locations according to the exact posterior probability of the location given all stations, but computing this posterior or sampling from it is intractable. A simpler but still effective proposer independently computes one predicted location per individual station, which is an easier problem, and then selects, according to a Gaussian mixture model, one of these predicted locations for being proposed. This is effective because, with enough stations, it is likely that one of the predictions will be close to the true location.

With this intuition, we use Bean Machine’s easily implementable proposer interface to write a custom proposer for the `event_attr` variable, which inspects the `det_attr(s)` children variables and uses a Gaussian mixture model proposer around the predicted attributes for each of the detections:

```py
class SingleSiteSeismicProposer(BaseSingleSiteMHProposer):

    def get_proposal_distribution(world: World) -> dist.Distribution:
        # self.node is given at initialization of the proposer since
        # there is one proposer per node
        node_var = world.get_variable(self.node)

        # instead of computing the probability of the event given
        # the join detections (a computation exponential in the number of detections),
        # we compute the probability of the event
        # given each separate detection and propose to use one of them
        # by sampling from a Gaussian mixture model.
        inverse_distributions_of_event =
            [inverted_laplace(world[child_det_attr])
            for child_det_attr in node_var.children]
        return create_gaussian_mixture_model(inverse_distributions_of_event)
```

Since this is a single site proposer, there is one proposer per node and each node is updated independently from all the others.
In this case we can just use `SingleSiteInference`, and run inference with that.
```py
SeismicInference = SingleSiteInference(SingleSiteSeismicProposer)
observations = {...}
samples = SeismicInference([event_attr()], observations, num_samples)
```

If we wanted more complex blocking logic, we would define our own inference class by overriding `get_proposers`.
Note that the user is free to define the blocking logic however they want; this is one example where we block all the variables together.

```py
class MultiSiteSeismicInference(BaseInference):

    def get_proposers(self, world, target_rvs, num_adaptive_samples):
        proposers = []
        for rv in target_rvs:
            proposers.append(SingleSiteSeismicProposer(rv))
        # This is a list of all the proposers to use during inference.
        # In this case, we will block everything together with SequentialProposer
        # This means that all of the proposers will be shuffled and proposed
        # in a single MH step.
        return [SequentialProposer(proposers)]

```

[1] Arora, Nimar & Russell, Stuart & Sudderth, Erik. (2013). NET-VISA: Network Processing Vertically Integrated Seismic Analysis. The Bulletin of the Seismological Society of America. 103. 709-729. 10.1785/0120120107.
