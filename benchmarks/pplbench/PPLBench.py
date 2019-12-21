# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

import argparse
import datetime
import importlib
import multiprocessing
import os
import pickle
import pkgutil
import random
import time
import traceback

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import models
import numpy as np
import pandas as pd
import ppls
import torch
import torch.tensor as tensor
from beanmachine.ppl.diagnostics.common_statistics import (
    effective_sample_size,
    split_r_hat,
)


# helper function(s)
def get_lists():
    models_list = [
        m.name[:-5]  # strip the "Model" suffix
        for m in pkgutil.iter_modules(models.__path__)
        if m.name.endswith("Model")
    ]
    ppls_list = [m.name for m in pkgutil.iter_modules(ppls.__path__)]
    return models_list, ppls_list


def exec_process(func, *args, **kwargs):
    """
    calls func(*args, **kwargs) in a separate process and returns the value
    """
    random_seed = int(torch.randint(2 ** 32, (1,)))

    def entry_point(conn, random_seed):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        conn.send(func(*args, **kwargs))
        conn.close()

    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.Process(target=entry_point, args=(child_conn, random_seed))
    p.start()
    retval = parent_conn.recv()
    p.join()
    return retval


def get_color_for_ppl(ppl):
    """
    Creates a mapping from ppl name to color
    Inputs:
    - ppl: name of ppl
    Outputs:
    - color for matplotlib
    """
    colors = {
        "nmc": "C0",
        "pyro": "C1",
        "stan": "C2",
        "jags": "C3",
        "bootstrapping": "C4",
        "beanmachine": "C6",
        "pymc3": "C7",
        "beanmachine-vectorized": "C8",
    }
    if ppl in colors:
        return colors[ppl]
    else:
        print(f"Please set color in get_color_for_ppl() in PPLBench.py for {ppl}")
        return "C9"


def generate_plot(posterior_predictive, args_dict):
    """
    Creates a plot for indicating sample posterior convergence for different PPLs
    Inputs:
    - posterior_predictive: posterior predictive log likelihoods of posterior samples
    - args_dict: arguments used for the benchmark run
    """
    K = args_dict["k"]
    N = int(args_dict["n"])
    trials = int(args_dict["trials"])
    train_test_ratio = float(args_dict["train_test_ratio"])

    plt.figure(figsize=(8, 6))
    plt.rcParams.update({"font.size": 18})
    plt.title(
        f'{args_dict["model"]} model \n'
        f"{int(N * train_test_ratio)} data-points"
        f"| {K} covariates | {trials} trials"
    )

    legend = []
    for ppl_name in posterior_predictive:
        ppl_data = tensor(posterior_predictive[ppl_name])
        samples = 1.0 + torch.arange(ppl_data.shape[1])

        avg_log = torch.cumsum(ppl_data, 1) / samples
        group_avg = torch.mean(avg_log, dim=0)
        group_min, _ = torch.min(avg_log, dim=0)
        group_max, _ = torch.max(avg_log, dim=0)
        ppl_color = get_color_for_ppl(ppl_name)
        label = args_dict[f"legend_name_{ppl_name}"]
        line, = plt.plot(samples, group_avg, color=ppl_color, label=label)
        plt.fill_between(
            samples, group_min, group_max, color=ppl_color, interpolate=True, alpha=0.3
        )
        legend.append(line)
    legend = sorted(legend, key=lambda line: line.get_label())
    plt.legend(handles=legend)
    ax = plt.axes()
    ax.set_xlabel("Samples")
    ax.set_ylabel("Average log predictive")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1e"))


def compute_trial_statistics(posterior_predictive):
    """
    Computes effective sample size per trial
    """
    stats = {}
    for ppl in posterior_predictive:
        ppl_data = tensor(posterior_predictive[ppl])
        num_trials = len(ppl_data)
        stats[ppl] = {}
        for t in range(num_trials):
            stats[ppl][t] = {}
            ppl_trial = ppl_data[t].unsqueeze(0)
            n_eff = effective_sample_size(ppl_trial)
            stats[ppl][t]["n_eff"] = n_eff.item()
            stats[ppl][t]["num_unique"] = len(torch.unique(ppl_trial))
    return stats


def combine_dictionaries(trial_info, timing_info):
    """
    Combines the timing info with the trial info
    """
    for ppl in trial_info:
        for trial in trial_info[ppl]:
            trial_info[ppl][trial].update(timing_info[ppl][trial])
    return trial_info


def compute_summary_statistics(posterior_predictive):
    """
    Computes r_hat and effective sample size (treating each trial as a chain)
    """
    stats = {}
    for ppl in posterior_predictive:
        stats[ppl] = {}
        ppl_data = tensor(posterior_predictive[ppl])
        stats[ppl]["r_hat"] = split_r_hat(ppl_data).item()
        stats[ppl]["n_eff"] = effective_sample_size(ppl_data).item()
    return stats


def get_sample_subset(posterior_predictive, args_dict):
    """
    Return subset of data uniformly drawn from logspace
    Inputs:
    - posterior_predictive: posterior predictive log likelihoods of posterior samples
    - args_dict: arguments used for the benchmark run

    Outputs:
    - sample_subset: a list of (ppl name, trial, sample, log_pred, avg_log_pred)
    """
    sample_subset = []
    subset_size = int(args_dict["plot_data_size"])
    num_samples = int(args_dict["num_samples"])
    num_trials = int(args_dict["trials"])
    log_space = np.logspace(0, np.log10(num_samples), num=subset_size, endpoint=False)
    indices = []
    for num in log_space:
        if int(num) not in indices:
            indices.append(int(num))

    for ppl in posterior_predictive:
        ppl_data = tensor(posterior_predictive[ppl])
        avg_log = torch.cumsum(ppl_data, 1) / (1.0 + torch.arange(ppl_data.shape[1]))
        for t in range(num_trials):
            for i in indices:
                log_pred = ppl_data[t][i].item()
                avg_log_pred = avg_log[t][i].item()
                sample_subset.append((ppl, t, i, log_pred, avg_log_pred))

    return sample_subset


def time_to_sample(ppl, module, runtime, data_train, args_dict, model=None):
    """
    Estimates the number of samples to run the inference given
    an expected runtime with a simple linear regression model:

    samples = alpha + beta * runtime

    inputs-
    module(PPL implementation module): the implementation to be timed
    runtime(seconds): how long to run the inference
    data_train(dict): training data
    args_dict: arguments

    returns-
    target_num_samples = number of posterior samples that would approximately
                  require that runtime
    """
    print("Estimating number of samples for given runtime...")
    args_dict[f"num_samples_{ppl}"] = 300
    args_dict[f"thinning_{ppl}"] = 1
    _, timing_info = module.obtain_posterior(
        data_train=data_train, args_dict=args_dict, model=model
    )

    if args_dict["include_compile_time"] == "yes":
        runtime = runtime - timing_info["compile_time"]
    time_for_300_samples = timing_info["inference_time"]
    args_dict[f"num_samples_{ppl}"] = 600
    _, timing_info = module.obtain_posterior(
        data_train=data_train, args_dict=args_dict, model=model
    )
    time_for_600_samples = timing_info["inference_time"]
    beta = 300 / (time_for_600_samples - time_for_300_samples)
    alpha = 0.5 * (
        (600 - beta * time_for_600_samples) + (300 - beta * time_for_300_samples)
    )
    target_num_samples = int(alpha + beta * runtime)
    if target_num_samples <= 0:
        print("ModelError:Target runtime too small; consider increasing it")
        exit(1)
    print(
        f"Inference requires {target_num_samples} samples"
        f" to run for {runtime} seconds"
    )
    return target_num_samples


def estimate_thinning(args_dict):
    min_num_samples = np.inf
    for p in (args_dict["ppls"]).split(","):
        ppls_args = p.split(":")
        if len(ppls_args) == 1:
            ppl = p
        else:
            ppl, _ = ppls_args
        if args_dict[f"num_samples_{ppl}"] <= min_num_samples:
            min_num_samples = args_dict[f"num_samples_{ppl}"]
    for p in (args_dict["ppls"]).split(","):
        ppls_args = p.split(":")
        if len(ppls_args) == 1:
            ppl = p
        else:
            ppl, _ = ppls_args
        args_dict[f"thinning_{ppl}"] = int(
            args_dict[f"num_samples_{ppl}"] / float(min_num_samples)
        )
        args_dict[f"num_samples_{ppl}"] = int(
            args_dict[f"thinning_{ppl}"] * min_num_samples
        )
    return args_dict


def get_args(models_list, ppls_list):
    # configure input arguments
    parser = argparse.ArgumentParser(
        description="PPLBench \
                                     - A Benchmark for Probabilistic Programming \
                                      Languages and Libraries"
    )
    parser.add_argument(
        "-m",
        "--model",
        default="robustRegression",
        help=str("Choose the model to benchmark. Options: " + str(models_list)),
    )
    parser.add_argument("-k", default="default", help="Number of covariates")
    parser.add_argument("-n", default="default", help="Size of simulated dataset")
    parser.add_argument(
        "-l",
        "--ppls",
        default="stan,jags",
        help=str("Choose the PPLs to benchmark. Options: " + str(ppls_list)),
    )
    parser.add_argument("--inference-type", default="mcmc", help="Options: mcmc, vi")
    parser.add_argument(
        "-t",
        "--runtime",
        default="default",
        help="estimated runtime (seconds/(model,ppl)",
    )
    parser.add_argument(
        "-s",
        "--num-samples",
        default=100,
        help="number of samples to sample from the posterior in \
        each trial of the inference algorithm",
    )
    parser.add_argument(
        "--model-args",
        default="default",
        help="model specific arguments; enter as comma-separated list",
    )
    parser.add_argument("--rng_seed", default=int(42))
    parser.add_argument(
        "--train-test-ratio",
        default="default",
        help="ratio to split simulated data into train and test sets",
    )
    parser.add_argument(
        "--trials", default="default", help="Number of times to re-run the test"
    )
    parser.add_argument(
        "--save-samples",
        default="no",
        help="if yes, samples will be stored in posterior_samples.csv",
    )
    parser.add_argument(
        "--save-generated-data",
        default="no",
        help="if yes, generated data stored in generated_data.csv",
    )
    parser.add_argument(
        "--include-compile-time",
        default="no",
        help="if yes, compile time will be considered in plots",
    )
    parser.add_argument(
        "--plot-data-size",
        default=100,
        help="number of samples per PPL per trial to save",
    )
    return parser.parse_args()


def save_data(
    args_dict,
    generated_data,
    posterior_samples,
    posterior_predictive,
    trial_info,
    summary_info,
    posterior_samples_subset,
):
    """
    Save data output folder
    Inputs:
    args_dict: arguments used for the benchmark run
    generated_data: data generated in benchmark run
    posterior_samples: posterior samples generated by PPLs in benchmark run
    posterior_predictive: posterior predictive log likelihoods of posterior samples
    trial_info: compile and inference times and n_eff for each trial for each PPL
    summary_info: effective sample size and r hat for each PPL
    posterior_samples_subset: subset of posterior samples for plot generation
    """

    with open(os.path.join(args_dict["output_dir"], "arguments.csv"), "w") as csv_file:
        pd.DataFrame.from_dict(args_dict, orient="index").to_csv(csv_file)

    if args_dict["save_generated_data"] == "yes":
        with open(
            os.path.join(args_dict["output_dir"], "generated_data.csv"), "w"
        ) as csv_file:
            pd.DataFrame.from_dict(generated_data, orient="index").to_csv(csv_file)

    if args_dict["save_samples"] == "yes":
        with open(
            os.path.join(args_dict["output_dir"], "posterior_samples.csv"), "w"
        ) as csv_file:
            pd.DataFrame.from_dict(posterior_samples, orient="index").to_csv(csv_file)

    with open(os.path.join(args_dict["output_dir"], "trial_info.csv"), "w") as csv_file:
        columns = [
            "ppl",
            "trial",
            "n_eff",
            "num_unique",
            "compile_time",
            "inference_time",
        ]
        data = []
        for ppl in trial_info:
            for trial in trial_info[ppl]:
                stats = []
                for col in columns[2:]:
                    stats.append(trial_info[ppl][trial][col])
                data.append((ppl, trial, *stats))
        pd.DataFrame(data, columns=columns).to_csv(csv_file, index=False)

    with open(
        os.path.join(args_dict["output_dir"], "summary_info.csv"), "w"
    ) as csv_file:
        columns = ["ppl", "r_hat", "n_eff"]
        data = []
        for ppl in summary_info:
            stats = []
            for col in columns[1:]:
                stats.append(summary_info[ppl][col])
            data.append((ppl, *stats))
        pd.DataFrame(data, columns=columns).to_csv(csv_file, index=False)

    with open(
        os.path.join(args_dict["output_dir"], "posterior_samples_subset.csv"), "w"
    ) as csv_file:
        columns = ["ppl", "trial", "sample", "log_predictive", "average_log_predictive"]
        pd.DataFrame(posterior_samples_subset, columns=columns).to_csv(
            csv_file, index=False
        )

    with open(
        os.path.join(args_dict["output_dir"], "posterior_predictives.pkl"), "wb"
    ) as f:
        pickle.dump(posterior_predictive, f)


def main():
    models_list, ppls_list = get_lists()
    args = get_args(models_list, ppls_list)
    args_dict = vars(args)
    # check is user passed model arguments, if yes parse them in an array as numbers
    if not args_dict["model_args"] == "default":
        args_dict["model_args"] = [
            eval(x) for x in (args_dict["model_args"]).split(",")
        ]
    # check if model exists, get model defaults for unspecified args
    if str(args.model) in models_list:
        model = importlib.import_module(str("models." + args.model + "Model"))
        defaults = model.get_defaults()
        for key in args_dict.keys():
            if args_dict[key] == "default":
                if key in defaults.keys():
                    args_dict[key] = defaults[key]
                else:
                    print(
                        "ModelError: no default found for", key, "in noisyOrTopicModel"
                    )
                    exit(1)
    else:
        print(args.model, "model not supported, exiting")
        exit(1)

    # generate model
    model_instance = model.generate_model(args_dict)

    # generate data
    generated_data = model.generate_data(args_dict=args_dict, model=model_instance)

    estimated_total_time = (
        1.1
        * int(args_dict["runtime"])
        * int(args_dict["trials"])
        * len(args_dict["ppls"].split(","))
    )

    print(
        "Starting benchmark; estimated time is"
        f" {int(estimated_total_time / 3600.)} hour(s),"
        f"{int((estimated_total_time % 3600) / 60)} minutes"
    )

    # Create timestamped folder outputs
    if not os.path.isdir(os.path.join(".", "outputs")):
        os.mkdir(os.path.join(".", "outputs"))
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime(
        "%d-%m-%Y_%H:%M:%S"
    )
    args_dict["output_dir"] = os.path.join(".", "outputs", timestamp)
    os.mkdir(args_dict["output_dir"])
    print(f"Outputs will be saved in : {args_dict['output_dir']}")

    # estimate samples for given runtime and decide thinning
    for p in (args.ppls).split(","):
        ppls_args = p.split(":")
        if len(ppls_args) == 1:
            ppl = p
            legend_name = p
        else:
            ppl, legend_name = ppls_args

        # check if the ppl has a corresponding model implementation module
        try:
            module = importlib.import_module(f"ppls.{ppl}.{args.model}")
        except ModuleNotFoundError:
            traceback.print_exc()
            print(f"{ppl} implementation not found for {args.model}; exiting...")
            exit()

        args_dict[f"legend_name_{ppl}"] = legend_name

        if args_dict["num_samples"] == 100:
            # estimate number of samples required for given time
            args_dict[f"num_samples_{ppl}"] = time_to_sample(
                ppl=ppl,
                module=module,
                runtime=int(args.runtime),
                data_train=generated_data["data_train"],
                args_dict=args_dict.copy(),
                model=model_instance,
            )
        else:
            args_dict[f"num_samples_{ppl}"] = int(args_dict["num_samples"])

    args_dict = estimate_thinning(args_dict)
    # obtain samples and evaluate predictvies
    timing_info = {}
    posterior_samples = {}
    posterior_predictive = {}
    for p in (args.ppls).split(","):
        ppls_args = p.split(":")
        if len(ppls_args) == 1:
            ppl = p
        else:
            ppl, _ = ppls_args
        # check if the ppl has a corresponding model implementation module
        try:
            module = importlib.import_module(f"ppls.{ppl}.{args.model}")
        except ModuleNotFoundError:
            continue
        print(f"{ppl}:")
        timing_info[ppl] = [None] * int(args_dict["trials"])
        posterior_samples[ppl] = [None] * int(args_dict["trials"])
        posterior_predictive[ppl] = [None] * int(args_dict["trials"])
        # start trial loop
        for i in range(int(args_dict["trials"])):
            print("Starting trial", i + 1, "of", args_dict["trials"])
            # obtain posterior samples and timing info
            posterior_samples[ppl][i], timing_info[ppl][i] = exec_process(
                module.obtain_posterior,
                data_train=generated_data["data_train"],
                args_dict=args_dict,
                model=model_instance,
            )
            # compute posterior predictive
            posterior_predictive[ppl][i] = exec_process(
                model.evaluate_posterior_predictive,
                samples=posterior_samples[ppl][i].copy(),
                data_test=generated_data["data_test"],
                model=model_instance,
            )
            print(
                f"Trial {i + 1} "
                f'complete in {timing_info[ppl][i]["inference_time"]} '
                "seconds.\n Statistics of  posterior predictive\n mean:"
                f"{np.array(posterior_predictive[ppl][i]).mean()}"
                f"\n var: {np.array(posterior_predictive[ppl][i]).var()}"
            )

    generate_plot(posterior_predictive, args_dict)
    plt.savefig(
        os.path.join(
            args_dict["output_dir"], "sample_posterior_convergence_behaviour.png"
        ),
        bbox_inches="tight",
        dpi=600,
    )

    posterior_samples_subset = get_sample_subset(posterior_predictive, args_dict)
    trial_info = compute_trial_statistics(posterior_predictive)
    trial_info = combine_dictionaries(trial_info, timing_info)
    summary_info = compute_summary_statistics(posterior_predictive)

    # save data
    save_data(
        args_dict,
        generated_data,
        posterior_samples,
        posterior_predictive,
        trial_info,
        summary_info,
        posterior_samples_subset,
    )
    print(f"Output in : {args_dict['output_dir']}")


if __name__ == "__main__":
    main()
