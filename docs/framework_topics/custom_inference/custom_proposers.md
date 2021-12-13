---
title: 'Custom Proposers'
sidebar_label: 'Proposers'
slug: '/custom_proposers'
---
## API

Bean Machine is flexible and easily allows users to supply custom Metropolis-Hastings proposers for specific random variables. This enables users to easily incorporate domain knowledge in inference.

For each iteration of sampling of variable $X$ with value $x$ using the Metropolis Hastings algorithm, we first sample a new value $x'$ for $X$ using proposal distribution $g$. We then update the world to reflect this change, and use the Metropolis Hastings ratio of
$$\text{min}\big[1, \frac{p(x')g(x \mid x')}{p(x)g(x' \mid x)}\big]$$
to decide whether to keep the update or to revert to the world as it was before it.

Bean Machine allows users to easily provide custom proposals for $g$, without needing to implement other details such as the calculations for $p(x')$ or the acceptance ratio. All proposers must inherit from the `AbstractSingleSiteProposer` class and implement the following methods:
```py
def propose(self, node: RVIdentifier, world: World) -> Tuple[Tensor, Tensor, Dict]:
```
```py
def post_process(
    self, node: RVIdentifier, world: World, auxiliary_variables: Dict
) -> Tensor:
```

### Propose
`propose` takes two parameters: the node and the world.
* node: $X$, the random variable to propose a new value for
* world: graphical data structure representing variable dependencies and values in the model *before* the proposal is made

The return value of `propose` is a tuple with three values
* $x'$, the new proposed value for the node (if a variable has an associated [transform](../custom_inference/transforms.md), this returned value must be in the variable's original space -- use `Variable.inverse_transform_value` to transform values in the transformed space back to the original space)
* $\log[g(x' \mid x)]$, the log probability of proposing this value
* a dictionary of auxiliary variables used in `propose` that are useful in `post_process` (this will typically be intermediary computations used by both functions, so they do not need to be recomputed by `post_process`)

### Post Process
`post_process` takes three parameters: the node, the world, and the auxiliary variables dictionary.
* node: (same as `propose`) $X$, the random variable to propose a new value for
* world: (same as `propose`, but *updated with the proposed* value for node) graphical data structure representing variable dependencies and values in the model
* auxiliary variables: the same dictionary returned by `propose`

The return value of `post_process` is $\log[g(x \mid x')]$, the log probability of proposing the original value given the new value.

<!--
    `post_process` it not a very descriptive name. Why not `reverse_proposal_distribution` or something similar?
-->

### Important Methods
The `node` is of type `RVIdentifier`, which includes only the function and its parameters. To access the `Variable` corresponding to this node in the world, call
```py
world.get_node_in_world_raise_error(node, False)
```
This returns the associated `Variable`.

<!--
    It is odd and surprising that world.get_node_in_world_raise_error(node, False) returns a Variable, not a node, in spite of its name.
    It might be good to rename it.
-->

## Sample Custom Proposer

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
class SingleSiteSeismicProposer(AbstractSingleSiteProposer):

    def value(self, child: RVIdentifier) -> Tensor:
        return world.get_node_in_world_raise_error(child, False).value

    def proposal_distribution_of_event_given_children_det_attr(event_node):
        # instead of computing the probability of the event given
        # the join detections (a computation exponential in the number of detections),
        # we compute the probability of the event
        # given each separate detection and propose to use one of them
        # by sampling from a Gaussian mixture model.
        event_var = world.get_node_in_world_raise_error(node, False)
        inverse_distributions_of_event_given_det_attrs =
            [inverted_laplace(self.value(child_det_attr))
            for child_det_attr in event_var.children]
        return create_gaussian_mixture_model(inverse_distributions_of_event_given_det_attrs)

    def propose(self, event_node: RVIdentifier, world: World) -> Tuple[Tensor, Tensor, Dict]:
        proposal_distribution = proposal_distribution_of_event_given_children_det_attr(event_node)
        new_value = proposal_distribution.sample()
        log_prob = proposal_distribution.log_prob(new_value)
        return new_value, log_prob, {}

    def post_process(self, event_node: RVIdentifier, world: World, aux_variables: Dict) -> Tensor:
        proposal_distribution = proposal_distribution_of_event_given_children_det_attr(event_node)
        old_value = world.get_old_value(event_node)
        return proposal_distribution.log_prob(old_value)
```

Note that the particular above proposer proposes a new value based solely on the values of the *children* of `node`, and does not use the *value* of node.
Therefore the value $\log[g(x' \mid x)]$ returned by `propose` is independent of the actual value $x$.
Likewise, the value $\log[g(x \mid x')]$ returned by `post_process` does not use the new value $x'$.

Because the proposal distribution is the same in both `propose` and `post_process` (since it only depends on the children variable which are not modified),
an important optimization in this sampler could be to add the value of `proposal_distribution` in the dictionary of auxiliary variables returned by `propose` (under a suitable key such as `'proposal_distribution'`) and simply reuse it in `post_process`:

```py
# last line of 'propose'
return new_value, log_prob, {'proposal_distribution': proposal_distribution}
```

```py
# first line of 'post_process'
proposal_distribution = aux_variables['proposal_distribution']
```

Also note that in `post_process` we must evaluate the proposal distribution on $x$, which is no longer available from `event_node.value`
because now the world has been updated with the new value $x'$.
However the old value is still available from `world.get_old_value(node)` (or `world.get_old_transformed_value(node)` if a transform is being used).


## Gradient-Based Proposers

The `World` class also provides support for gradient-based proposers (see World and Variable API documentation for more details). For all continuous variables, the gradient is tracked for `transformed_value`. Because of the locally-structured nature of Bean Machine models, the contribution of a variable to a world's likelihood is determined solely from the values of the variable itself and its child nodes (those nodes whose distributions are directly influenced by the value of the variable). The likelihood of the relevant parts of the world, also known as the score, can be computed using
```py
compute_score(node_var: Variable) -> Tensor
```
To calculate the gradient of a variable in the `propose` method, we first need to access the `Variable` object corresponding to the `RVIdentifier` node. We then compute the score of node, and then take the gradient with respect to the `transformed_value` using the `grad` implementation supported by PyTorch.

```py
node_var = world.get_node_in_world(node, False)
score = world.compute_score(node_var)
first_gradient = grad(score, node_val)
```
The gradient can now be used to obtain a new proposal for the variable.

[1] Arora, Nimar & Russell, Stuart & Sudderth, Erik. (2013). NET-VISA: Network Processing Vertically Integrated Seismic Analysis. The Bulletin of the Seismological Society of America. 103. 709-729. 10.1785/0120120107.
