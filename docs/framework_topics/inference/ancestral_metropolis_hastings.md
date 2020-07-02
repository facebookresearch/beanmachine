**Single-Site Ancestral MH**

In Single-Site Ancestral Metropolis Hastings, we update the values of random variables one variable at a time (hence the name “Single-Site”).

There are four main steps in Single Site Ancestral Metropolis-Hastings.


1. Propose a value for node A. The proposed value is sampled from the node’s prior distribution.
2. Given the proposed value, we update the distributions of all nodes in the Markov blanket of node A. The Markov blanket of a node consists of the node’s children, node’s parents, and the other parents of the node’s children. We only consider the Markov blanket of node A because the distributions of these nodes change when the value of node A changes. All other nodes in the network are independent of node A given the nodes in the Markov blanket of node A.
3. Compute the log proposal ratio of proposing this new value given the updated distributions of all nodes in the Markov blanket of node A, using the standard Metropolis Hastings ratio.
4. Accept or reject the proposed value with the probability computed in Step 3.


Here is an example of how to use Single Site Ancestral Metropolis Hastings to perform inference in Bean Machine.

```py
from beanmachine.ppl.inference.single_site_ancestral_mh import SingleSiteAncestralMetropolisHastings

mh = SingleSiteAncestralMetropolisHastings()
coin_samples = mh.infer(queries, observations, num_samples, num_chains, run_in_parallel)
```
```queries ```: List of random variables that we want to get posterior samples for
```observations```: Dict, where key is the random variable, and value is the value of the random variable
```num_samples```: number of samples to run inference for
```num_chains```: number of chains to run inference for
```run_in_parallel```: True if you want the chains to run in parallel

For an example of Single Site Metropolis Hastings in a simple coin toss model, check out our tutorials!
