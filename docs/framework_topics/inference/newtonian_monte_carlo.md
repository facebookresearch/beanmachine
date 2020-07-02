# Newtonian Monte Carlo

Newtonian Monte Carlo (NMC) is a second-order gradient-based Markov Chain Monte Carlo (MCMC) algorithm that uses the first and second gradients to propose a new value for a random variable. In NMC, if we are at x for random variable X, we propose the next value by:

1. Computing the first and second gradient of the joint density with respect to X at x.
2. Drawing the next sample, x’, from the forward proposer, Q(x’ | X = x) = N (x − ∇2 log π(x) ^(−1) ∇ log π(x), −∇2 log π(x) ^(−1)).
3. Computing the probability of proposing x’ using the forward proposer: Q(x’ | X = x)
4. Computing the first and second gradient of the joint density with respect to X at x’.
5. Computing the probability of proposing x using the reverse proposer: Q(x | X = x’) = N (x’ − ∇2 log π(x’) ^(−1) ∇ log π(x’), −∇2 log π(x’) ^(−1)), Q(x | X = x’).
6. Computing the Metropolis Hasting acceptance probability by computing: [π(x’) * Q(x | X = x’)]/ [π(x) * Q(x’ | X = x)]

Single site Metropolis Hasting is the fundamental algorithm in Bean Machine’s inference engine making second gradients tractable. 

Bean Machine’s NMC provides:

* Half-space proposer for random variables defined only over positive reals.
* Simplex proposer for random variables defined only over reals from [0, 1]
* Real-space proposer for all continuous random variables. Bean Machine uses transformation techniques to transform bounded continuous random variables to unbounded continuous random variables. 


