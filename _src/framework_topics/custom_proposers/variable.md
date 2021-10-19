## Worlds and Variables

A crucial part of Bean Machine inference is its ability track the state of inference using `World`s. During inference, the `World` represents the state of the variables and their dependencies. The `World` is a graphical data structure where each node in the graph represents a `Variable` containing information about the variable's value, distribution, probability, etc. The edges between nodes represent the dependencies between variables. Each variable tracks its parent variables as well as its children variables.


#### RVIdentifier

`RVIdentifier` is a dataclass used to identify the random variable and does not contain any of the inference information. It only consists of the variable's `function` and `arguments` as supplied in the model declaration.

#### Variable

The `Variable` represents the instantiation of each random variable in a model. Each `RVIdentifier` has a corresponding `Variable`. It has the following attributes:

* `distribution: Distribution` - the prior distribution denoted in the modeled
* `value: Tensor` - the value of the Variable at the current state of inference
* `parent: Set[Optional[RVIdentifier]]` - the set of random variables called within the function declaration of the random variable
* `children: Set[Optional[RVIdentifier]]` - the set of random variables which call this variable in their function declaration
* `log_prob: Tensor` - the log probability of the value with the prior distribution
* `proposal_distribution: ProposalDistribution` - the proposal distribution used during inference
* `is_discrete: bool` - an indicator of whether it is a discrete variable as opposed to a continuous variable
* `transform: Transform` - a Transform to be applied in order to reshape the state space, see Transform documentation
* `transformed_value: Tensor` - the value of the variable in the transformed space. This value will be the same as `value` when no transforms are specified.
* `jacobian: Tensor` - the log Jacobian determinant of the transforms

When writing custom proposers, the two most relevant functions within the `Variable` API are
```py
transform_value(self, value: Tensor) -> Tensor
inverse_transform_value(self, transformed_value: Tensor) -> Tensor
```
Because Bean Machine expects the proposal to be a value in its original space, it may be helpful to use the `variable.transformed_value` throughout inference before converting back to the original space using `inverse_transform_value`.

#### World

The `World` tracks the variables through the `WorldVars` class, which associates each `RVIdentifier` to a `Variable`. When new values are proposed to the `World` and the world is updated to reflect these changes, these changes are stored in the world's `DiffStack`, which is a stack of `Diff`s, each tracking Variables with differing values probabilities, parents, children, etc.

When writing custom proposers for random variables, it is often important to access the `Variable` corresponding to the `RVIdentifier` to see the current value, sample from the variable's prior distribution, etc. This can be done through the function
```py
get_node_in_world(node: RVIdentifier, to_be_copied = False, to_create_new_diff = False)
```
* `to_be_copied`: a flag indicating if the world should copy this variable to the `DiffStack` and start tracking changes. Within the `proposal` method of a proposer, this should generally be set to False, as the world will automatically add the proposed value for the Variable to a `Diff` once the proposal method is completed.
* `to_create_new_diff`: whether an additional diff with these changes should be added to the top of the `DiffStack`. This should generally be set to False.
