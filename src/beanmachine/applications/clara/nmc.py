# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch
from torch import tensor
from torch.distributions import Categorical, Dirichlet
from tqdm import tqdm


def simplex_proposer(val, grad):
    """
    propose for a simplex-constrained random variable
    """
    conc = grad * val + 1
    assert (conc > 0).all()
    return Dirichlet(conc)


class State:
    """
    Represents state of system.

    static State consists of:
    - num_labelers: the number of labelers assigning labels
    - num_categories: the number of possible categories for each item and its label
    - items: item indices
    - labels: assigned labels
    - labelers: labelers corresponding to assigned labels

    dynamic state consists of:
    - prevalence: the probability that an item is of a category (size num_categories)
    - true_label: the true label of each item (size num_items)
    - confusion: for each labeler and true label the probability of picking
    a category (size num_labelers x num_categories x num_categories)

    """

    num_labelers = 0
    num_categories = 0
    num_items = 0
    items = None
    labels = None
    labelers = None

    @classmethod
    def init_class(
        cls,
        num_labelers,
        num_categories,
        expected_correctness,
        concentration,
        labels,
        labelers,
        num_labels,
    ):
        cls.num_labelers = num_labelers
        cls.num_categories = num_categories
        cls.num_items = len(num_labels)
        items = []
        for idx, num in enumerate(num_labels):
            items.extend([idx] * num)
        cls.items = tensor(items)
        cls.labels = labels
        cls.labelers = torch.LongTensor(labelers)
        cls.true_label_proposer = Categorical(
            torch.ones(cls.num_categories) / cls.num_categories
        )
        cls.prevalence_prior = Dirichlet(
            torch.ones(cls.num_categories) / cls.num_categories
        )
        conc = torch.zeros((cls.num_categories, cls.num_categories))
        conc[:, :] = (
            concentration * (1 - expected_correctness) / (cls.num_categories - 1)
        )
        conc[torch.arange(cls.num_categories), torch.arange(cls.num_categories)] = (
            concentration * expected_correctness
        )
        cls.confusion_prior = Dirichlet(conc)
        cls.uniform_0_1 = torch.distributions.Uniform(0, 1)

    def __init__(self):
        self.prevalence = torch.ones(State.num_categories) / State.num_categories
        self.true_labels = State.true_label_proposer.sample([State.num_items])
        self.confusion = (
            torch.ones((State.num_labelers, State.num_categories, State.num_categories))
            / State.num_categories
        )

    def update_(self):
        self.update_prevalence()
        self.update_labels_()
        self.update_confusion_()

    def update_prevalence(self):
        curr_score, proposer = self.propose_prevalence(
            self.prevalence_prior, self.prevalence, self.true_labels
        )
        new_prevalence = proposer.sample()
        new_score, new_proposer = self.propose_prevalence(
            self.prevalence_prior, new_prevalence, self.true_labels
        )
        log_acc = (
            new_score
            - curr_score
            + new_proposer.log_prob(self.prevalence)
            - proposer.log_prob(new_prevalence)
        )
        if log_acc > self.uniform_0_1.sample().log():
            self.prevalence = new_prevalence

    def update_labels_(self):
        # uniform proposer
        new_labels = self.true_label_proposer.sample(self.true_labels.shape)
        # prior
        log_acc = (
            self.prevalence[new_labels].log() - self.prevalence[self.true_labels].log()
        )
        # likelihood
        log_acc.scatter_add_(
            0,
            self.items,
            self.confusion[self.labelers, new_labels[self.items], self.labels].log()
            - self.confusion[
                self.labelers, self.true_labels[self.items], self.labels
            ].log(),
        )
        changed = log_acc > self.uniform_0_1.sample(log_acc.shape).log()
        self.true_labels[changed] = new_labels[changed]

    def update_confusion_(self):
        score, proposer = self.propose_labeler_label_confusion(self.confusion)
        new_confusion = proposer.sample()
        new_score, new_proposer = self.propose_labeler_label_confusion(new_confusion)
        log_acc = (
            new_score
            - score
            + new_proposer.log_prob(self.confusion)
            - proposer.log_prob(new_confusion)
        )
        changed = log_acc > self.uniform_0_1.sample(log_acc.shape).log()
        self.confusion[changed] = new_confusion[changed]

    def propose_labeler_label_confusion(self, confusion):
        # we will convert the true labels of the labeled items into 1-hot encoded
        true_one_hot = torch.zeros((len(self.items), self.num_categories))
        true_one_hot.scatter_(1, self.true_labels[self.items].unsqueeze(1), 1)
        # score the current value
        confusion = confusion.clone().requires_grad_(True)
        # prior for each labeler, true_label
        score = self.confusion_prior.log_prob(confusion)
        # likelihood
        score.scatter_add_(
            0,
            self.labelers.unsqueeze(1).expand(-1, self.num_categories),
            confusion[self.labelers, self.true_labels[self.items], self.labels]
            .unsqueeze(1)
            .log()
            * true_one_hot,
        )
        ssum = score.sum()
        (grad,) = torch.autograd.grad(
            ssum, confusion, retain_graph=True, create_graph=True
        )
        # simulate the cost of computing a vector Hessian
        for _ in range(State.num_categories):
            torch.autograd.grad(ssum, confusion, retain_graph=True)
        grad = grad.detach()
        confusion.requires_grad_(False)
        proposer = Dirichlet(grad * confusion + 1)
        return score, proposer

    def propose_prevalence(self, prior, prevalence, labels):
        prevalence = prevalence.clone().requires_grad_(True)
        score = prior.log_prob(prevalence) + prevalence[labels].log().sum()
        (grad,) = torch.autograd.grad(
            score, prevalence, retain_graph=True, create_graph=True
        )
        # simulate the cost of computing a vector Hessian
        for _ in range(State.num_categories):
            torch.autograd.grad(score, prevalence, retain_graph=True)
        grad = grad.detach()
        prevalence.requires_grad_(False)
        return score, simplex_proposer(prevalence, grad)


def obtain_posterior(data_train, args_dict, model=None):
    """
    NMC implementation of crowdsourced annotation model

    Inputs:
    - data_train(tuple of np.ndarray): vector_y, vector_J_i, num_labels
    - args_dict: a dict of model arguments
    Returns:
    - samples(dict): posterior samples of all parameters
    - timing_info(dict): compile_time, inference_time
    """
    num_samples = args_dict["num_samples_nmc"]

    burn_in = args_dict["burn_in_nmc"] if "burn_in_nmc" in args_dict.keys() else 0
    n_categories, _, expected_correctness, concentration = args_dict["model_args"]
    # first parse out the data into static fields in the State class
    compile_time_t1 = time.time()
    State.init_class(
        int(args_dict["k"]),
        n_categories,
        expected_correctness,
        concentration,
        *data_train
    )
    curr = State()
    compile_time_t2 = time.time()
    # repeatedly update the state
    infer_time_t1 = time.time()
    samples = []
    for i in tqdm(range(burn_in + num_samples), desc="inference"):
        curr.update_()
        if i < burn_in:
            continue
        samples_dict = {
            "theta": curr.confusion.clone().numpy(),
            "pi": curr.prevalence.clone().numpy(),
        }
        samples.append(samples_dict)
    infer_time_t2 = time.time()
    timing_info = {
        "compile_time": compile_time_t2 - compile_time_t1,
        "inference_time": infer_time_t2 - infer_time_t1,
    }
    return (samples, timing_info)
