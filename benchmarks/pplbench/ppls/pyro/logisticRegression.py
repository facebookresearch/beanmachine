import time

import pyro
import pyro.contrib.autoguide as autoguide
import pyro.distributions as dist
import pyro.infer
import pyro.optim
import torch
from pyro.infer import TracePredictive


def logistic_model(x_train, y_train):
    K = int(x_train.shape[1])
    scale_alpha, scale_beta, loc_beta = model_args

    alpha = pyro.sample("alpha", dist.Normal(0.0, scale_alpha))
    beta = pyro.sample("beta", dist.Normal(torch.ones(K) * loc_beta, scale_beta))
    mu = alpha + x_train @ beta
    return pyro.sample("y", dist.Bernoulli(logits=mu), obs=y_train)


def wrapped_model(x_train, y_train):
    pyro.sample("prediction", dist.Delta(logistic_model(x_train, y_train)))


def obtain_posterior(data_train, args_dict, model=None):
    """
    Pyro implementation of logistic regression model.

    Inputs:
    - data_train(tuple of np.ndarray): x_train, y_train
    - args_dict: a dict of model arguments
    Returns:
    - samples_jags(dict): posterior samples of all parameters
    - timing_info(dict): compile_time, inference_time
    """
    assert pyro.__version__.startswith("0.4.1")
    global model_args
    model_args = args_dict["model_args"]
    LEARNING_RATE = 1e-2
    NUM_STEPS = 100

    # x_train is (num_features, num_observations)
    x_train, y_train = data_train
    # x_train is (num_observations, num_features)
    x_train = x_train.T
    # convert them to tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    start_time = time.time()
    # the guide serves as an approximation to the posterior
    guide = autoguide.AutoDiagonalNormal(logistic_model)
    optimiser = pyro.optim.Adam({"lr": LEARNING_RATE})
    loss = pyro.infer.Trace_ELBO()

    # set up the inference algorithm
    svi = pyro.infer.SVI(
        logistic_model,
        guide,
        optimiser,
        loss,
        num_samples=args_dict["num_samples_pyro"],
    )
    pyro.clear_param_store()

    # do the gradient steps
    for _step in range(NUM_STEPS):
        svi.step(x_train, y_train)

    elapsed_time_sample_pyro = time.time() - start_time

    # repackage samples into shape required by PPLBench
    samples = []
    posterior = svi.run(x_train, y_train)

    # posterior predictive distribution we can get samples from
    trace_pred = TracePredictive(
        wrapped_model, posterior, num_samples=args_dict["num_samples_pyro"]
    ).run(x_train, None)
    trace = trace_pred.exec_traces
    for i in range(args_dict["num_samples_pyro"]):
        sample_dict = {}
        sample_dict["alpha"] = trace[i].nodes["alpha"]["value"].detach().numpy()
        sample_dict["beta"] = trace[i].nodes["beta"]["value"].detach().numpy()
        samples.append(sample_dict)
    timing_info = {
        "compile_time": 0,
        "inference_time": elapsed_time_sample_pyro,
    }  # no compiliation for pyro
    return (samples, timing_info)
