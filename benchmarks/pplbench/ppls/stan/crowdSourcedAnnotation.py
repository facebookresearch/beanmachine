# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import pickle
import time

import pystan


CODE = """
data {
    int<lower=1> n_obs;  // len(vector_y)
    int<lower=1> n_items;  // I
    int<lower=1> n_labelers;  // J
    int<lower=1> n_categories;  // K
    real<lower=0> expected_correctness;
    real<lower=0> concentration;
    int<lower=1> n_labels[n_items];  // num_labels
    int<lower=1, upper=n_categories> labels[n_obs];  // vector_y
    int<lower=1, upper=n_labelers> labeler[n_obs]; // vector_J_i
}

transformed data {
  vector[n_categories] beta;
  vector[n_categories] alpha[n_categories];
  beta = rep_vector(1./n_categories, n_categories);
  alpha = rep_array(rep_vector(concentration * (1-expected_correctness)
                               / (n_categories-1), n_categories), n_categories);
  for (k in 1:n_categories) {
    alpha[k,k] = concentration * expected_correctness;
  }
}

parameters {
  // pi(Category): The true probabilities of each category.
  simplex[n_categories] pi;
  // theta: confusion matrix
  simplex[n_categories] theta[n_labelers,n_categories];
}

model {
  int pos;
  int a;
  int label;
  real inner_prod;
  real lp;
  pi ~ dirichlet(beta);
  for (j in 1:n_labelers) {
    for (k in 1:n_categories) {
      theta[j,k,:] ~ dirichlet(alpha[k,:]);
    }
  }
  pos = 1;
  for (i in 1:n_items) {
    lp = 0.0;
    for (k in 1:n_categories) {
      inner_prod = 0.0;
      for (j in 0:(n_labels[i] - 1)) {
        label = labels[pos + j];
        a = labeler[pos + j];
        inner_prod += categorical_lpmf(label | theta[a, k]);
      }
      lp += pi[k] * exp(inner_prod);
    }
    target += log(lp);
    pos += n_labels[i];
  }
}
"""


def obtain_posterior(data_train, args_dict, model=None):
    """
    Stan implementation of crowdsourced annotation model

    Inputs:
    - data_train(tuple of np.ndarray): vector_y, vector_J_i, num_labels
    - args_dict: a dict of model arguments
    Returns:
    - samples_stan(dict): posterior samples of all parameters
    - timing_info(dict): compile_time, inference_time
    """
    global CODE
    vector_y, vector_J_i, num_labels = data_train
    n_labelers = int(args_dict["k"])
    n_items = len(num_labels)
    n_categories, _, expected_correctness, concentration = args_dict["model_args"]

    data_stan = {
        "n_obs": len(vector_y),
        "n_items": n_items,
        "n_labelers": n_labelers,
        "n_categories": n_categories,
        "expected_correctness": expected_correctness,
        "concentration": concentration,
        "n_labels": num_labels,
        "labels": [i + 1 for i in vector_y],
        "labeler": [i + 1 for i in vector_J_i],
    }  # Stan uses 1 indexing

    code_loaded = None
    pkl_filename = os.path.join(
        args_dict["output_dir"], "stan_crowdSourcedAnnotation.pkl"
    )
    if os.path.isfile(pkl_filename):
        model, code_loaded, elapsed_time_compile_stan = pickle.load(
            open(pkl_filename, "rb")
        )
    if code_loaded != CODE:
        # compile the model, time it
        start_time = time.time()
        model = pystan.StanModel(model_code=CODE, model_name="crowd_sourced_annotation")
        elapsed_time_compile_stan = time.time() - start_time
        # save it to the file 'model.pkl' for later use
        with open(pkl_filename, "wb") as f:
            pickle.dump((model, CODE, elapsed_time_compile_stan), f)

    # sample the parameter posteriors, time it
    start_time = time.time()
    if args_dict["inference_type"] == "mcmc":
        fit = model.sampling(
            data=data_stan, iter=int(args_dict["num_samples_stan"]), chains=1
        )
    elif args_dict["inference_type"] == "vi":
        fit = model.vb(data=data_stan, iter=args_dict["num_samples_stan"])
    samples_stan = fit.extract(pars=["theta", "pi"], permuted=False, inc_warmup=True)
    elapsed_time_sample_stan = time.time() - start_time

    # repackage samples into shape required by PPLBench
    samples = []
    for i in range(int(args_dict["num_samples_stan"])):
        sample_dict = {}
        for parameter in samples_stan.keys():
            sample_dict[parameter] = samples_stan[parameter][i]
            # theta samples have shape (1, J, K, K), need to squeeze the 1 out
            if sample_dict[parameter].shape[0] == 1:
                sample_dict[parameter] = sample_dict[parameter].squeeze()
        samples.append(sample_dict)
    timing_info = {
        "compile_time": elapsed_time_compile_stan,
        "inference_time": elapsed_time_sample_stan,
    }

    return (samples, timing_info)
