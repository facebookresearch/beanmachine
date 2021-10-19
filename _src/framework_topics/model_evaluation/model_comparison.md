# Model comparison

Let's suppose you have a problem for which you have several candidate models. How do you determine which is best? You can run several diagnostics on the posterior samples. However, more directly, you could assess performance on held-out test data!

To do this, you can simulate data from the generative model and split it into training and test data. We then want to run inference on the model using the training data, but assess prediction accuracy on the test data. Specifically, we use something called "predictive log likelihood" as our measure of predictive accuracy. Predictive log likelihood asks how likely the _test_ data was, using random variables inferred by running inference on the _training_ data. We actually use a _log_ probability by convention, so this takes on a value in $(-\infty, 0]$. Predictive log likelihood ($\ell_\text{pred}$) is defined as

\[
    \ell_\text{pred} = \log \left( \frac{1}{n} \sum_{i=1}^n \mathbb{P}(x_\text{test} \mid z_i) \right),
\]

where $z_i$ represents the assignment of values to random variables in inference iteration $i \in [1, N]$, and $\mathbb{P}(x_\text{test} \mid z_i)$ is the probability of the test data being generated if $z_i$ were the values of the random variables in the model.

You can then generate predictive log likelihood plots ($\ell_\text{pred}$ against samples) for several models using test data to compare models:

1. We can compare the final value of the predictive log likelihood plots and the model with a higher pll explains the data better.
2. Predictive log likelihood plots can also indicate a notion of speed of convergence. We can compare the rate at which log likelihood increases with number of samples.
