import itertools
import os
import sys
import time

import numpy as np
import torch
from julia.api import Julia


def round_pmf_vector(ls):
    d = 5
    ls = np.clip(ls, a_min=(10 ** -d), a_max=1 - (10 ** -d)).astype(np.float32)
    ls = ls.round(d)
    # Using strings is the key step, for some reason.
    ls = [float(elem) for elem in ls.astype(str)]
    ll = len(ls)
    delta = 1.0 - sum(ls)
    for i in range(ll):
        if ls[i] > 2 * abs(delta):
            ls[i] += delta
            break
    return np.array(ls)


def square_matrix(ls, k):
    # assert len(ls) == k ** 2
    # Need to unpack single-element lists at bottom level of PyJulia result
    return [list(itertools.chain(*ls[i * k : (i + 1) * k])) for i in range(k)]


def obtain_posterior(data_train, args_dict, model=None):
    """
    Gen impmementation of HMM prediction.

    :param data_train:
    :param args_dict: a dict of model arguments
    :returns: samples_beanmachine(dict): posterior samples of all parameters
    :returns: timing_info(dict): compile_time, inference_time
    """
    # true_theta = model["theta"]
    # true_theta = torch.stack(true_theta).numpy()

    N = int(args_dict["n"])
    K = int(args_dict["k"])
    concentration, mu_loc, mu_scale, sigma_shape, sigma_scale, observe_model = list(
        map(float, args_dict["model_args"])
    )
    num_samples = int(args_dict["num_samples"])
    observations = list(torch.stack(data_train).numpy())

    start_time = time.time()
    """
    Argument compiled_modules necessary for PyJulia on Conda+Ubuntu:
    # https://pyjulia.readthedocs.io/en/latest/troubleshooting.html
    #your-python-interpreter-is-statically-linked-to-libpython
    """
    jl = Julia() # compiled_modules=False)
    elapsed_time_compile = time.time() - start_time

    # Assemble julia code
    if not observe_model:
        print("Dirichlet distribution not supported by Gen")
        return

    # This use of sys.path assumes that the PPLBench.py is being executed
    jl_filename = os.path.join(
        sys.path[0], "ppls/gen/hiddenMarkov" + str(bool(observe_model)) + ".jl"
    )
    jl_code = "".join(open(jl_filename, "r").readlines())
    # jl_code += "\n Random.seed!(" + str(args_dict["rng_seed"]) + ")"

    jl_arglist = [
        N,
        K,
        concentration,
        mu_loc,
        mu_scale,
        sigma_shape,
        sigma_scale,
        num_samples,
        observations,
    ]
    if observe_model:
        tht = np.array([round_pmf_vector(t) for t in model["theta"].numpy()]).tolist()
        jl_arglist += [
            # list(itertools.chain(*model["theta"].numpy())),
            tht,
            model["mus"].numpy().tolist(),
            model["sigmas"].numpy().tolist(),
        ]
    jl_argstring = ",".join([str(s) for s in jl_arglist])
    jl_code += "\n return main(model, " + jl_argstring + ")"
    """
    For example..

    if observe_model == False:
    main(20,3,0.1,1.0,5.0,3.0,3.0,46,[-3.5, -7.1, 5.7, -4.0, -3.6, 4.5, -4.5,
    -4.8, 7.8, -4.5, -3.6, 10.2, -4.4, -4.9, 5.7, -5.0, -2.9, 7.3, -4.3, -3.2])

    if observe_model == True:
    main(20,3,0.1,1.0,5.0,3.0,3.0,46, [-3.5, -7.1, 5.7, -4.0, -3.6, 4.5, -4.5,
    -4.8, 7.8, -4.5, -3.6, 10.2, -4.4, -4.9, 5.7, -5.0, -2.9, 7.3, -4.3, -3.2],
    ,[[0.0001249760389328003, 0.17351700365543365, 0.8263580203056335],
    [0.0016700197011232376, 0.00901000015437603, 0.9893199801445007],
    [0.5541480183601379, 0.011416000314056873, 0.43443599343299866]],
    [1.031515121459961, -2.329209089279175, 0.10338902473449707],
    [1.8671684265136719, 0.38424789905548096, 0.7053596377372742])
    """
    start_time = time.time()
    posterior_samples = jl.eval(jl_code)
    elapsed_time_sample = time.time() - start_time

    # Format final results.
    xn1str = "X[" + str(N - 1) + "]"

    if observe_model:
        xs = posterior_samples  # [-1]
        samples_formatted = []
        for xn1 in xs:
            samples_formatted.append({xn1str: xn1 - 1})

    else:
        xs, mus, sigmas = posterior_samples
        # Want to swap the way these are ordered, so we can iterate through.
        # Such that, e.g., thetas[i] gives the 'i'th MCMC sample of theta as a list.
        mus = [[musK[i] for musK in mus] for i in range(num_samples)]
        sigmas = [[sigmasK[i] for sigmasK in sigmas] for i in range(num_samples)]
        xn1s = xs[-1]
        # Now, reformat each theta[i] to be 2-dimensional square matrix.

        samples_formatted = []
        for mu, sigma, xn1 in zip(mus, sigmas, xn1s):
            result = {}
            # Need to unpack single-element lists at bottom level of PyJulia result
            # result["mus"] = sum(mus, [])
            result["mus"] = list(itertools.chain(*mu))
            result["sigmas"] = list(itertools.chain(*sigma))
            result[xn1str] = xn1 - 1
            samples_formatted.append(result)

    timing_info = {
        "compile_time": elapsed_time_compile,
        "inference_time": elapsed_time_sample,
    }

    return (samples_formatted, timing_info)
