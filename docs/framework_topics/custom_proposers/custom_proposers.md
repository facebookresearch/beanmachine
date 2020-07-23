## Custom Proposers

Bean Machine is flexible and easily allows users to supply custom Metropolis Hastings proposers for specific random variables. The enables users to easily incorporate domain knowledge to inference.

For each iteration of sampling of variable $X$ with value $x$ using the Metropolis Hastings algorithm, we first sample a new value $x'$ for $X$ using proposal distribution $g$. We then update the world to reflect this change, and use the Metropolis Hastings ratio of
$$\text{min}\big[1, \frac{p(x')g(x \mid x')}{p(x)g(x' \mid x)}\big]$$
to decide whether or not to update the value of $X$ from $x$ to $x'$.

Bean Machine allows users to easily provide custom proposals for $g$, without needing to implement other details such as the calculations for $p(x')$ or the acceptance ratio. All proposers must inherit from the `AbstractSingleSiteProposer` class and implement the following methods:
```py
def propose(self, node: RVIdentifier, world: World) -> Tuple[Tensor, Tensor, Dict]:
```
```py
def post_process(
    self, node: RVIdentifier, world: World, auxiliary_variables: Dict
) -> Tensor:
```

#### Propose
`propose` takes two parameters: the node and the world.
* node: $X$, the random variable to propose a new value for
* world: graphical data structure representing variable dependencies and values in the model

The return value of `propose` is a tuple with three values
* $x'$, the new proposed value for the node
* $\log[g(x' \mid x)]$, the log probability of proposing this value
* a dictionary of the auxiliary variables used in propose that are needed by `post_process`

#### Post Process
`post_process` take three parameters: the node, the world, and the auxiliary variables dictionary.
* node: (same as `propose`) $X$, the random variable to propose a new value for
* world: (same as `propose`) graphical data structure representing variable dependencies and values in the model
* auxiliary_variables: the same dictionary returned by `propose`

The return value of `post_process` is $\log[g(x \mid x')]$, the log probability of proposing the original value given the new value.

### Important Methods
The `node` is of type `RVIdentifier`, which includes only the function and its parameters. To access the `Variable` corresponding to this node in the world, call
```py
world.get_node_in_world_raise_error(node, False)
```
This returns the associated `Variable` (see Variable API documentation for more details).

### Sample Custom Proposer

Here, we implement a custom proposer for locating seismic events on Earth as described in [?]. In particular, the simplified version of the model is used, where one seismic event, denoted in the model by `event_attr()`, occurs randomly on the surface of the Earth. This event sends seismic energy in a single wave. There are different seismic stations across the surface of the Earth, and each station will noisily record its attributes of time, azimuth, and slowness, denoted by `det_attr(s)`. The inference problem is to find the `event_attr()` given the `det_attr(s)`.

```py
@bm.random_variable
def event_attr():
  return SeismicEventPrior()

@bm.random_variable
def det_attr(station):
    det_loc = calculate_detection(station, event_attr())
    return Laplace(det_loc, scale)
```

There is domain knowledge within seismology to mathematically solve for the most likely attributes of an event given the detection attributes for a specific station. Due to the noisy nature of the data, these predicted locations can be inaccurate. However, with enough stations, it is likely that one of the predictions will be close to the true values. With this intuition, we used Bean Machine’s easily implementable proposer interface to write a custom proposer for the `event_attr` variable, which inspects the `det_attr(s)` children variables and uses a Gaussian mixture model proposer around the predicted attributes for each of the detections.

```py
class SingleSiteSeismicProposer(AbstractSingleSiteProposer):
    def propose(self, node: RVIdentifier, world: World) -> Tuple[Tensor, Tensor, Dict]:
        node_var = world.get_node_in_world_raise_error(node, False)
        inverted_detections = []
        for child in node_var.children:
            child_var = world.get_node_in_world_raise_error(child, False)
            child_det_attr = child_var.value
            inverted_detection = invert(child_det_attr)
            inverted_detections.append(inverted_detection)
        gmm = create_gausian_mixture_model(inverted_detections)
        proposal = gmm.sample()
        log_prob = gmm.log_prob(proposal)
        return proposal, log_prob, {}

    def post_process(
        self, node: RVIdentifier, world: World, auxiliary_variables: Dict
    ) -> Tensor:      
        node_var = world.get_node_in_world_raise_error(node, False)
        current_value = node_var.value
        # compute gmm as before
        return gmm.log_prob(current_value)
```

### Gradient-based proposers

The `World` also provides support for gradient-based proposers (see World and Variable API documentation for more details). For all continuous variables, the gradient is tracked for `transformed_value`. Because of the single-site nature of Bean Machine, the only part of the world's likelihood which related to a variable originate from the variable itself and its child nodes, which have distributions dependent on the value of the variable. The likelihood of the relevant parts of the world, also known as the score, can be computed using
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
