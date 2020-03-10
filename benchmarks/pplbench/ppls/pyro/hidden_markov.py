import time
from typing import Dict, List, Tuple

import pyro
import pyro.distributions as dist
import torch
import torch.tensor as tensor
from pyro.infer.mcmc import MCMC, NUTS


def hmm_model(K, N, observations, model_args, model):
    X = {}
    Y = {}
    concentration, mu_loc, mu_scale, sigma_shape, sigma_scale, observe_model = list(
        map(float, model_args)
    )
    with pyro.plate("data", K):
        if observe_model:
            theta = pyro.sample(
                "theta",
                dist.Dirichlet(torch.ones(K) * concentration / K),
                obs=model["theta"],
            )
            mus = pyro.sample("mus", dist.Normal(mu_loc, mu_scale), obs=model["mus"])
            sigmas = pyro.sample(
                "sigmas", dist.Normal(sigma_shape, sigma_scale), obs=model["sigmas"]
            )
        else:
            theta = pyro.sample(
                "theta", dist.Dirichlet(torch.ones(K) * concentration / K)
            )
            mus = pyro.sample("mus", dist.Normal(mu_loc, mu_scale))
            sigmas = pyro.sample("sigmas", dist.Normal(sigma_shape, sigma_scale))
    X[0] = pyro.sample(
        "X[0]", dist.Categorical(tensor([1.0] + [0.0] * (K - 1))), obs=tensor(0)
    )
    Y[0] = pyro.sample(
        "Y[0]", dist.Normal(mus[X[0]], sigmas[X[0]]), obs=observations[0]
    )
    for i in pyro.markov(range(1, N)):
        X[i] = pyro.sample(
            "X[" + str(i) + "]",
            dist.Categorical(theta[X[i - 1]]),
            infer={"enumerate": "parallel"},
        )
        Y[i] = pyro.sample(
            "Y[" + str(i) + "]",
            dist.Normal(mus[X[i]], sigmas[X[i]]),
            obs=observations[i],
        )


def obtain_posterior(
    data_train: List, args_dict: Dict, model=None
) -> Tuple[List, Dict]:
    """
    Pyro impmementation of HMM prediction.

    :param data_train:
    :param args_dict: a dict of model arguments
    :returns: samples_numpyro(dict): posterior samples of all parameters
    :returns: timing_info(dict): compile_time, inference_time
    """

    N = int(args_dict["n"])
    K = int(args_dict["k"])
    num_samples = int(args_dict["num_samples_pyro"])
    num_warmup = 50
    observations = data_train
    # observations = torch.stack(observations).numpy()
    concentration, mu_loc, mu_scale, sigma_shape, sigma_scale, observe_model = list(
        map(float, args_dict["model_args"])
    )

    assert num_samples - num_warmup > 0

    # Run NUTS
    start_time = time.time()
    nuts_kernel = NUTS(model=hmm_model)
    mcmc = MCMC(
        kernel=nuts_kernel,
        num_samples=num_samples - num_warmup,
        warmup_steps=num_warmup,
        num_chains=1,
    )
    mcmc.run(
        K=K,
        N=N,
        observations=observations,
        model_args=args_dict["model_args"],
        model=model,
    )
    samples_pyro = mcmc.get_samples()
    elapsed_time_sample_numpyro = time.time() - start_time
    # repackage samples into shape required by PPLBench
    samples = []

    for i in range(num_samples):
        result = {}
        if not observe_model:
            result["theta"] = samples_pyro["theta"][i].numpy()
            result["mus"] = samples_pyro["mus"][i]
            result["sigmas"] = samples_pyro["sigmas"][i]

        # NOTE: Pyro doesn't have values for discrete latent states (X's)
        # so assigning everything to 1 so it doesn't break.
        # We can use this model to get timing info, not evaluation info.
        result["X[" + str(N - 1) + "]"] = 1
        samples.append(result)
    timing_info = {"compile_time": 0, "inference_time": elapsed_time_sample_numpyro}
    print("inference time: ", elapsed_time_sample_numpyro)

    return (samples, timing_info)
