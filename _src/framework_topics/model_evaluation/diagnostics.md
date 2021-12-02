---
id: diagnostics
title: 'Model Evaluation in Bean Machine: Diagnostics Module'
sidebar_label: 'Model Evaluation in Bean Machine: Diagnostics Module'
slug: '/diagnostics'
---

This notebook introduces the Diagnostics class in Bean Machine (BM) which aims to assist the modeler to get insights about the model performance. Diagnostics currently supports two main components: 1) General Summary Statistics module and 2) Visualizer module.

**General Summary Statistics module** aggregates the statistics of all or a subset of queries over a specific chain or all chains.

**Visualizer module** processes samples and encapsulates the results in a plotly object which could be used for actual visualization.


Both of the BM Diagnostics components support function registration which allows the user to extend each component with new functionalities that modeler might be interested to have.

The rest of this document goes over examples of how each component can be called or extended.


# Model definition and running inference:

Suppose we want to use BM Diagnostics module over the inferred samples of the following model:

```python
Example:

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

# Summary Stats

Calling the summary() function on learned model parameters outputs a table including mean, standard deviation (std), confidence interval (CI), r-hat and effective sample size (n-eff).

User also has the flexibility to define new functions and register them as part of the available summary stat functions. So, when summary() is called, the results for the user defined function will be added to the output table.

Here are different ways that modeler can call summary() function:

## 1. Getting the full statistics summary of all queries over all chains

Simply call summary() to get a comprehensive table of gathered statistics for all queries.

```python
out_df= Diagnostics(samples).summary()
```

## 2. Getting summary statistics for a subset of queries

Considering a big list of queries that model may have, user may hand-pick a subset of queries to confine the output table.

```python
Example:

out_df = Diagnostics(samples).summary([dirichlet(1, 5), beta(0)])
```

## 3. Getting summary for a specific chain

To compare how results may change over the course of chains, user can pass the chain number to summary() function.

```python
Example:

out_df = Diagnostics(samples).summary(query_list=[dirichlet(1, 5)], chain=1)

```
## 4. Extend summary module by new functions:

User has the option to extend the summary module by registering new functions or overwrite an already available functions. To add new functions, user should make a derived class of the Diagnostics class and register new functions in the class constructor as follow:

```python

def newfunc(query_samples: Tensor) -> Tensor

class Mydiag(Diagnostics):
    def __init__(self, samples: MonteCarloSamples):
        super().__init__(samples)
        self.newfunc = self.summaryfn(newfunc, display_names=["new_func"])
```

**summaryfn** wrapper treats the newfunc as part of the summary module and adds its results to the output table.

The input to all default summary stat functions has the shape of **torch.Size([C, N])** where **C** and **N** are the number of chains and samples-per-chain respectively considering that **query_samples** is as follow:

```python
samples = mh.infer(...)
query_samples = samples[query] ## query_samples.shape = torch.Size([C, N])
```

```python
Example:

def mymean(query_samples: Tensor) -> Tensor:
    return torch.mean(query_samples, dim=[0, 1])


class Mydiag(Diagnostics):
    def __init__(self, samples: MonteCarloSamples):
        super().__init__(samples)
        self.mymean = self.summaryfn(mymean, display_names=["mymean"])


customDiag = Mydiag(samples)
out = customDiag.summary(query_list=[dirichlet(1, 5)], chain=0)
out.head()
```

## 5. Individual calling of summary statistics functions:
```python
Example:

# calling user-defined func over all chains
out_df = customDiag.mymean([dirichlet(1, 5)])

# calling user-defined func over a specific chain
out_df = customDiag.mymean([dirichlet(1, 5)], chain = 1)

# calling a default func over a specific chain
out_df = customDiag.std([dirichlet(1, 5)], chain = 1)

```

## 6. Override an already registered function

```python
Example:

def mean(query_samples: Tensor) -> Tensor:
    y = query_samples + torch.ones(query_samples.size())
    return torch.mean(y, dim=[0, 1])

class Mydiag(Diagnostics):
    def __init__(self, samples: MonteCarloSamples):
        super().__init__(samples)
        self.mean = self.summaryfn(mean, display_names=["avg"])

customDiag = Mydiag(samples)
out = customDiag.summary(query_list=[dirichlet(1, 5)], chain=0)
```

# Visualization:
Currently we support trace plots and auto-correlation plots for samples of a requested model parameter. User can define new visualization-related functions and register them via **plotfn** wrapper. Each of these functions return a plotly object and so user can modify the object as he/she wishes. Here are different ways to call plot over whole or a subset of queries.

## 1. Execute all plot-related functions for all queries
User can enable plotting the returned plotly object by passing display = True to the plot() function. The default is false which means that only the plotly object is returned. So, user has the flexibility to update the layout for the outputted object and frame the outputted plot they way he/she wishes.

```python
Example:
fig = Diagnostics(samples).plot()
```

## 2. Execute all plot-related functions for a subset of queries

```python
Example:
figs = Diagnostics(samples).plot(query_list=[dirichlet(1, 5)])
```


## 3. Update and display the plotly object

```python
Example:
for _i,fig in enumerate(figs):
    fig.update_layout(paper_bgcolor="LightBlue",height=1500, width=700,)
    fig.update_layout(legend_orientation="h")
    fig.update_xaxes(title_font=dict(size=14, family='Courier', color='crimson'))
    fig.update_yaxes(title_font=dict(size=14, family='Courier', color='crimson'))
    plotly.offline.iplot(fig)
```
## 4. Execute all plot-related functions for a specific chain

```python
Example:
d = Diagnostics(samples)
figs = d.plot(query_list=[dirichlet(1, 5)], chain = 0)
```
## 5. Individual calling of a plot-related function:


```python
Example:
d = Diagnostics(samples)

autocorr_object = d.autocorr([dirichlet(1, 5)]) # pass "display = True" to output the plot
autocorr_object = d.autocorr([dirichlet(1, 5)], chain = 0)
```
