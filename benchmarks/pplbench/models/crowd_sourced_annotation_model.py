# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Module for Crowd-Sourced Annotation Model

Paper describing this: https://www.aclweb.org/anthology/Q14-1025

Module contains following methods:
get_defaults()
generate_model()
generate_data()
evaluate_posterior_predictive()

Model Specification:
This model attempts to find the true label of an item based on labels
provided by a set of imperfect labelers.
There are I items, J labelers, and K label classes. Each item i is labelled by
|J_i| labelers
z_i : true label of item i
y_ij : label provided to item i by labeler j

pi_k : probability of label class k
theta_jkk' = the probability that labeler j will label an item of true class k
             with label k'

for all labelers J, theta_jkk' has a dirichilet prior alpha_k
alpha_k = concentration * {
    expected_correctness for k == k',
    (1-expected_correctness)/(K-1) for k != k'
}

pi has a Dirichlet(beta) prior with beta_k = 1/K for all k
pi ~ Dirichlet(beta)

z_i ~ Categorical(pi)

for j in J:
    for k in K:
        theta_jk ~ Dirichlet(alpha_k)

for i in I:
    |J_i| ~ Poisson(labeler_rate)
    J_i = |J_i| labelers chosen at random from J without replacement

for i in I:
    for j in J_i:
        y_ij ~ Categorical(theta_jz_i)

Inputs to PPL implementations: y_ij_train, J_i of size I/2
Samples obtained from PPL Implementation: pi*, theta*

posterior predictive log likelihood computation (ppll):
ppll = 0
for i in I/2 to I:
    P_i = 0
    for k in K:
        P_k = 1
        for j in J_i:
            P_k *= P(y_ij_test| theta*_jk)
        P_i += pi*_k * P_k
    ppll += log(P_i)

Model specific arguments:
global arg -n -> number of items (model arg I)
global arg -k -> number of lablers (model arg J)
Pass these arguments in following order -
[num_label_classes(K), labeler_rate, expected_correctness, concentration]
"""

import numpy as np
from tqdm import tqdm


def vector_index_form(y, J_i):
    """
    helper function to convert 'ragged array' data structure of y and J_i to a
    vector-index form
    Inputs:
    y, J_i: ragged arrays of labels and labelers
    Outputs:
    vector_y, vector_J_i : flattened vectors of y and J_i
    num_labels : number of labels in rach row
    """
    vector_y = []
    vector_J_i = []
    num_labels = []
    for i in range(len(y)):
        vector_y.extend(y[i])
        vector_J_i.extend(J_i[i])
        num_labels.append(len(y[i]))
    return (vector_y, vector_J_i, num_labels)


def get_defaults():
    """
    Returns model defaults for Crowd-Sourced Annotation Model
    """
    defaults = {
        "n": 500,  # number of items to be labelled
        "k": 50,  # number of labellers
        "runtime": 200,
        "train_test_ratio": 0.5,
        "trials": 10,
        "model_args": [3, 2.5, 0.5, 10.0],
    }
    return defaults


def generate_model(args_dict):
    """
    This model does not require instatiation of model structure
    """
    return None


def generate_data(args_dict, model=None):
    """
    Generate data for Crowd-Sourced Annotation Model.
    Inputs:
    - N(int) = number of items I
    - K(int) = number of lablers J
    - num_label_classes = number of label classes K
    - labler_rate = possion rate for number of lablers per item

    Returns:
    - generated_data(dict) =
        - data_train: y_ij_train and J_i_train (labels and labelers)
        - data_test: y_ij_test and J_i_test (labels and labelers)
    """
    # load args
    print("Generating data")
    J = int(args_dict["k"])
    n_items = int(args_dict["n"])
    K, labeler_rate, expected_correctness, concentration = args_dict["model_args"]
    train_test_ratio = float(args_dict["train_test_ratio"])

    # choose a true class z for each item
    beta = 1 / K * np.ones(K)
    pi = np.random.dirichlet(beta)  # shape [K]
    z = np.random.choice(range(K), p=pi, size=n_items)  # shape [I]
    # set prior that each labeler on average has 50% chance of getting true label
    alpha = ((1 - expected_correctness) / (K - 1)) * np.ones([K, K]) + (
        expected_correctness - (1 - expected_correctness) / (K - 1)
    ) * np.eye(K)
    alpha *= concentration
    # sample confusion matrices theta for labelers from this dirichlet prior
    theta = np.zeros([J, K, K])
    for j in range(J):
        for k in range(K):
            # theta_jk ~ Dirichlet(alpha_k)
            theta[j, k] = np.random.dirichlet(alpha[k])

    # select labelers for each item, get their labels for that item
    J_i = []
    y = []
    i = 0
    while i < n_items:
        num_j = np.random.poisson(labeler_rate)
        # check if we sampled 0, if yes redo this loop
        if num_j == 0:
            continue
        J_i.append(np.random.choice(J, size=num_j, replace=False))
        y_i = []
        for j in J_i[i]:
            y_i.append(np.random.choice(range(K), p=theta[j, z[i]]))
        i += 1
        y.append(y_i)
    # split into train test and return
    split = int(train_test_ratio * n_items)
    data_train = vector_index_form(y[:split], J_i[:split])
    return {"data_train": data_train, "data_test": (y[split:], J_i[split:])}


def evaluate_posterior_predictive(samples, data_test, model=None):
    """
    Computes the likelihood of held-out testset wrt parameter samples

    input-
    samples(dict): parameter samples from model pi* and theta*
    data_test: test data y_ij_test

    returns- pred_log_lik_array(np.ndarray): log-likelihoods of data
            wrt parameter samples
    """
    """
    posterior predictive log likelihood computation (ppll):
    ppll = 0
    for i in I/2 to I:
        P_i = 0
        for k in K:
            P_k = 1
            for j in J_i:
                P_k *= P(y_ij_test| theta*_jk)
            P_i += pi*_k * P_k
        ppll += log(P_i)
    """
    y, J_i = data_test
    n_items = len(y)
    K = samples[0]["theta"].shape[1]
    pred_log_lik_array = []
    for sample in tqdm(samples, desc="eval", leave=False):
        sample_pred_log_lik = 0
        for i in range(n_items):
            P_i = 0
            for k in range(K):
                P_k = 1
                for index, j in enumerate(J_i[i]):
                    P_k *= sample["theta"][j, k, y[i][index]]
                P_i += sample["pi"][k] * P_k
            sample_pred_log_lik += np.log(P_i)
        pred_log_lik_array.append(sample_pred_log_lik)
    # return as a numpy array
    return np.array(pred_log_lik_array)
