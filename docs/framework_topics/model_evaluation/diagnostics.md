---
id: diagnostics
title: 'Diagnostics Module'
sidebar_label: 'Diagnostics Module'
slug: '/diagnostics'
---

:::warning
The Diagnostics module is still available but will be deprecated in future releases in favor of [ArviZ](https://arviz-devs.github.io/arviz/). Check out the tutorials for examples of Arviz integration.
:::

This section describes the Diagnostics class in Bean Machine (BM) which aims to assist the modeler to get insights about the model performance. Diagnostics currently supports two main components:

**General Summary Statistics Module**: aggregates the statistics of all or a subset of queries over a specific chain or all chains.

**Visualizer Module**: processes samples and encapsulates the results in a [Plotly](https://plotly.com/python/) graphing library object which could be used for actual visualization.


Both of the BM Diagnostics components support function registration which allows the user to extend each component with new functionalities that modeler might be interested to have.

The rest of this document goes over examples of how each component can be called or extended by showing how they would be used after defining the following model and running inference on it:


```python
@sample
def dirichlet(i, j):
    return dist.Dirichlet(
    torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [2.0, 3.0, 1.0]])
)


@sample
def beta(i):
    return dist.Beta(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([9.0, 8.0, 7.0]))


@sample
def normal():
    return dist.Normal(torch.tensor([0.0, 1.0, 2.0]), torch.tensor([0.5, 1.0, 1.5]))


mh = SingleSiteAncestralMetropolisHastings()
chains = 2
samples = mh.infer([beta(0), dirichlet(1, 5), normal()], {}, 50, chains)
```

## Summary Statistics

Calling the `summary()` method on a sample set outputs a table including mean, standard deviation (`std`), confidence interval (`CI`), $\hat{R}$ (`r-hat`),effective sample size (`n-eff`), as well as user-defined functions (as shown below).

Here are different ways to invoke the `summary()` function:

### 1. Getting Summary Statistics for All Queries over All Chains

Simply call `summary()` to get a comprehensive table of gathered statistics for all queries.

```python
out_df= Diagnostics(samples).summary()
```

### 2. Getting Summary Statistics for a Subset of Queries

Considering a large number of queries that the model may have, the user may hand-pick a subset of queries to confine the output table to.

```python
out_df = Diagnostics(samples).summary([dirichlet(1, 5), beta(0)])
```

### 3. Getting Summary for a Specific Chain

To compare how results may change over the course of a particular chain, the user can pass the chain number to the `summary()` method.

```python
out_df = Diagnostics(samples).summary(query_list=[dirichlet(1, 5)], chain=1)

```
### 4. Extending Summary with New Functions

The user has the option to extend the summary module by registering new functions or overwrite an already available function. To add new functions, they must define a sub-class of `Diagnostics` and register new functions in the class' `__init__` function.

The input to these user-defined functions must be the samples for a particular query, that is, a tensor of shape `torch.Size([C, N])` where `C` and `N` are the number of chains and samples-per-chain respectively. The samples for a query can typically be obtained by code like the following:

```python
samples = mh.infer(...)
query_samples = samples[query] ## query_samples.shape == torch.Size([C, N])
```

The following illustrates how to add a user-defined mean function to a `Diagnostics` subclass:

```python
def my_mean(query_samples: Tensor) -> Tensor:
    return torch.mean(query_samples, dim=[0, 1])


class MyDiag(Diagnostics):
    def __init__(self, samples: MonteCarloSamples):
        super().__init__(samples)
        self.my_mean_stat = self.summaryfn(my_mean, display_names=["My mean"])


custom_diag = MyDiag(samples)
out = custom_diag.summary(query_list=[dirichlet(1, 5)], chain=0)
```

### 5. Invoking Individual Summary Statistics Functions

```python
# Obtaining user-defined statistic over all chains
out_df = custom_diag.my_mean_stat([dirichlet(1, 5)])

# Obtaining user-defined statistic over one chain
out_df = custom_diag.my_mean_stat([dirichlet(1, 5)], chain = 1)

# Obtaining default statistic over one chain
out_df = custom_diag.std([dirichlet(1, 5)], chain = 1)

```

### 6. Overriding an Already Registered Function

Instead of defining a new summary statistic, the user can override one of the default ones. Here the user redefines `mean` by using the following line in the `__init__` function:

```python
self.mean = self.summaryfn(my_mean, display_names=["avg"])
```

## Visualization

Currently we support trace plots and auto-correlation plots for samples of a requested model parameter. The user can extend this by defining new visualization functions returning a [Plotly](https://plotly.com/python/) object and registering them via `plotfn` method, analogously to the way `summaryfn` was used for defining new summary statistics.

Here are different ways to call plot over all, or a subset of, queries.

### 1. Execute All Plot-Related Functions for All Queries

To compute all Plotly objects given a set of samples, simply invoke the `plot()` method. The method returns the Plotly objects without displaying them. To display them as well, use a `display=True` argument.

```python
fig = Diagnostics(samples).plot()  # returns a Plotly object
fig = Diagnostics(samples).plot(display=True)  # returns a Plotly object and displays it
```

### 2. Execute All Plot-Related Functions for a Subset of Queries

```python
figs = Diagnostics(samples).plot(query_list=[dirichlet(1, 5)])
```

### 3. Update and Display the Plotly Object

```python
for _i,fig in enumerate(figs):
    fig.update_layout(paper_bgcolor="LightBlue",height=1500, width=700,)
    fig.update_layout(legend_orientation="h")
    fig.update_xaxes(title_font=dict(size=14, family='Courier', color='crimson'))
    fig.update_yaxes(title_font=dict(size=14, family='Courier', color='crimson'))
    plotly.offline.iplot(fig)
```

### 4. Execute All Plot-Related Functions for a Specific Chain

```python
figs = Diagnostics(samples).plot(query_list=[dirichlet(1, 5)], chain = 0)
```

### 5. Individual Calling of a Plot-Related Function

```python
d = Diagnostics(samples)

autocorr_object = d.autocorr([dirichlet(1, 5)]) # pass "display = True" to output the plot
autocorr_object = d.autocorr([dirichlet(1, 5)], chain = 0)
```
