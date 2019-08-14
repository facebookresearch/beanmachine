# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

import argparse
import csv
import datetime
import importlib
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# helper function(s)
def get_lists():
    # Dynamically populate Models and PPLs available
    models_list = [
        x[:-8] for x in next(os.walk("./models"))[2] if not x.startswith("__")
    ]
    # NOTE: x[:-8] removes 'Model.py' from filename; to make it easy to use later
    # in calling PPL Implementations of this model.
    ppls_list = [x for x in next(os.walk("./ppls"))[1] if not x.startswith("__")]
    return models_list, ppls_list


def logspace_datadump(averaged_pp_list, time_axis_list, plot_data_size):
    """
    Create a datadump which uniformly subsamples in logspace
    Inputs:
    - averaged_pp_list: 2D array of average posterior predictive log-likelihoods
                        with shape [iterations, samples]
    - time_axis_list: list of timestamps corresponding to each sample
                      in averaged_pp_list, shape [iterations, samples]
    - plot_data_size(int): size of the subsampled arrays averaged_pp_list and time_axis

    Outputs:
    - data_dict: 'iterations': logspace subsampled iterations of averaged_pp_list
                 'time_axes':  logspace subsampled time axes
                 'log_indices': list of indices which are uniform in logspace.
    """
    data_dict = {}
    log_indices = []
    log_pre_index = np.logspace(
        0, np.log10(time_axis_list.shape[1]), num=plot_data_size, endpoint=False
    )
    for j in log_pre_index:
        if not int(j) in log_indices:
            log_indices.append(int(j))
    data_dict["log_indices"] = log_indices
    data_dict["time_axes"] = time_axis_list[:, log_indices]
    data_dict["iterations"] = averaged_pp_list[:, log_indices]
    return data_dict


def generate_plots(timing_info, posterior_predictive, args_dict):
    """
    Generates plots of posterior predictive log-likelihood

    inputs-
    timing_info(dict): 'PPL'->'compile_time' and 'inference_time'
    posterior_predictive(dict): 'PPL'->'iteration'->array of posterior_predictives

    returns-
    plt - a matplotlib object with the plots
    plt_data - a dict containing data for generating plots
    """
    K = args_dict["k"]
    N = args_dict["n"]
    plt_data = {}
    for PPL in posterior_predictive.keys():
        averaged_pp_list = []
        time_axis_list = []
        thinning = args_dict[f"thinning_{PPL}"]
        for iteration in range(len(posterior_predictive[PPL])):
            # timing_info[PPL][iteration] contains 'inference_time'
            # and 'compile_time' for each iteration of each PPL
            sample_time = timing_info[PPL][iteration]["inference_time"]
            # posterior_predictive[PPL][iteration] is a 1D array of
            # posterior_predictives for a certain PPL for certain iteration
            averaged_pp = np.zeros_like(posterior_predictive[PPL][iteration])
            time_axis = np.linspace(
                start=0, stop=sample_time, num=len(posterior_predictive[PPL][iteration])
            )
            if args_dict["include_compile_time"] == "yes":
                time_axis += timing_info[PPL][iteration]["compile_time"]
            for i in range(len(posterior_predictive[PPL][iteration])):
                averaged_pp[i] = np.mean(posterior_predictive[PPL][iteration][: i + 1])
            averaged_pp_list.append(averaged_pp)
            time_axis_list.append(time_axis)
        # plot!
        plt.xlim(left=0.01, right=time_axis[-1])
        plt.grid(b=True, axis="y")
        plt.xscale("log")
        plt.title(
            f'{args_dict["model"]} model \n'
            f"{N} data-points | {K} covariates | {iteration + 1} iterations"
        )
        averaged_pp_list = np.array(averaged_pp_list)
        time_axis_list = np.array(time_axis_list)
        max_line = np.max(averaged_pp_list, axis=0)
        min_line = np.min(averaged_pp_list, axis=0)
        mean_line = np.mean(averaged_pp_list, axis=0)
        mean_time = np.mean(time_axis_list, axis=0)
        label = f"{PPL}, {averaged_pp_list.shape[1] * thinning} samples/iteration"
        plt.plot(mean_time, mean_line, label=label)
        plt.fill_between(
            mean_time, y1=max_line, y2=min_line, interpolate=True, alpha=0.3
        )
        plt_data[PPL] = logspace_datadump(
            averaged_pp_list, time_axis_list, int(args_dict["plot_data_size"])
        )
    plt.legend()
    plt.ylabel("Average log predictive")
    plt.xlabel(
        "Time(Seconds)",
        "NOTE: includes compile time"
        if args_dict["include_compile_time"] == "yes"
        else None,
    )

    return plt, plt_data


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
        if runtime <= 50:
            print("ModelError:Target runtime too small; consider increasing it")
            exit()
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
    print(
        f"Inference requires {target_num_samples} samples"
        f" to run for {runtime} seconds"
    )
    return target_num_samples


def estimate_thinning(args_dict):
    min_num_samples = np.inf
    for ppl in (args_dict["ppls"]).split(","):
        if args_dict[f"num_samples_{ppl}"] <= min_num_samples:
            min_num_samples = args_dict[f"num_samples_{ppl}"]
    for ppl in (args_dict["ppls"]).split(","):
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
        "--iterations", default="default", help="Number of times to re-run the test"
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
        help="number of samples per PPL per iteration to save",
    )
    return parser.parse_args()


def save_data(
    args_dict,
    generated_data,
    posterior_samples,
    posterior_predictive,
    timing_info,
    timestamp,
    plot_data,
):
    """
    Save data output folder
    Inputs:
    args_dict: arguments used for the benchmark run
    generated_data: data generated in benchmark run
    posterior_samples: posterior samples generated by PPLs in benchmark run
    posterior_predictive: posterior predictive log likelihoods of posterior samples
    timing_info: timing information of each PPL, i.e. the compile and inference times
    timestamp: timestamp used as folder name to store outputs
    plot_data: data structure that stores information to recreate the plots
    """
    with open(f"./outputs/{timestamp}/plot_data.csv", "w") as csv_file:
        csvwriter = csv.writer(csv_file, delimiter=",")
        csvwriter.writerow(
            ["ppl", "iteration", "sample", "average_log_predictive", "time"]
        )
        for ppl in plot_data:
            for iteration in range(plot_data[ppl]["iterations"].shape[0]):
                for sample in range(len(plot_data[ppl]["iterations"][iteration, :])):
                    csvwriter.writerow(
                        [
                            ppl,
                            iteration + 1,
                            plot_data[ppl]["log_indices"][sample]
                            * args_dict[f"thinning_{ppl}"],
                            plot_data[ppl]["iterations"][iteration, sample],
                            plot_data[ppl]["time_axes"][iteration, sample],
                        ]
                    )

    with open(f"./outputs/{timestamp}/arguments.csv", "w") as csv_file:
        pd.DataFrame.from_dict(args_dict, orient="index").to_csv(csv_file)

    if args_dict["save_generated_data"] == "yes":
        with open(f"./outputs/{timestamp}/generated_data.csv", "w") as csv_file:
            pd.DataFrame.from_dict(generated_data, orient="index").to_csv(csv_file)

    if args_dict["save_samples"] == "yes":
        with open(f"./outputs/{timestamp}/posterior_samples.csv", "w") as csv_file:
            pd.DataFrame.from_dict(posterior_samples, orient="index").to_csv(csv_file)

    with open(f"./outputs/{timestamp}/timing_info.csv", "w") as csv_file:
        pd.DataFrame.from_dict(timing_info, orient="index").to_csv(csv_file)

    with open(f"./outputs/{timestamp}/posterior_predictives.pkl", "wb") as f:
        pickle.dump(posterior_predictive, f)


def main():
    models_list, ppls_list = get_lists()
    args = get_args(models_list, ppls_list)
    args_dict = vars(args)
    # check is user passed model arguments, if yes parse them in an array as numbers
    if not args_dict["model_args"] == "default":
        args_dict["model_args"] = [
            float(x) for x in (args_dict["model_args"]).split(",")
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
        * int(args_dict["iterations"])
        * len(args_dict["ppls"].split(","))
    )

    print(
        "Starting benchmark; estimated time is"
        f" {int(estimated_total_time / 3600.)} hour(s),"
        f"{int((estimated_total_time % 3600) / 60)} minutes"
    )

    # Create timestamped folder outputs
    if not os.path.isdir("./outputs"):
        os.mkdir("./outputs")
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime(
        "%d-%m-%Y_%H:%M:%S"
    )
    os.mkdir(f"./outputs/{timestamp}")
    print(f"Outputs will be saved in : /outputs/{timestamp}/")

    # estimate samples for given runtime and decide thinning
    for ppl in (args.ppls).split(","):
        # check if the ppl has a corresponding model implementation module
        if os.path.isfile(f"./ppls/{ppl}/{args.model}.py"):
            # import ppl module
            module = importlib.import_module(f"ppls.{ppl}.{args.model}")
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
            print(f"{ppl} implementation not found for {args.model}; exiting...")
            exit()
    args_dict = estimate_thinning(args_dict)
    # obtain samples and evaluate predictvies
    timing_info = {}
    posterior_samples = {}
    posterior_predictive = {}
    for ppl in (args.ppls).split(","):
        # check if the ppl has a corresponding model implementation module
        if os.path.isfile(f"./ppls/{ppl}/{args.model}.py"):
            print(f"{ppl}:")
            timing_info[ppl] = [None] * int(args_dict["iterations"])
            posterior_samples[ppl] = [None] * int(args_dict["iterations"])
            posterior_predictive[ppl] = [None] * int(args_dict["iterations"])
            # import ppl module
            module = importlib.import_module(f"ppls.{ppl}.{args.model}")
            # start iteration loop
            for i in range(int(args_dict["iterations"])):
                print("Starting iteration", i + 1, "of", args_dict["iterations"])
                # obtain posterior samples and timing info
                posterior_samples[ppl][i], timing_info[ppl][
                    i
                ] = module.obtain_posterior(
                    data_train=generated_data["data_train"],
                    args_dict=args_dict,
                    model=model_instance,
                )
                # compute posterior predictive
                posterior_predictive[ppl][i] = model.evaluate_posterior_predictive(
                    samples=posterior_samples[ppl][i].copy(),
                    data_test=generated_data["data_test"],
                    model=model_instance,
                )
                print(
                    f"Iteration {i + 1} "
                    f'complete in {timing_info[ppl][i]["inference_time"]} '
                    "seconds.\n Statistics of  posterior predictive\n mean:"
                    f"{np.array(posterior_predictive[ppl][i]).mean()}"
                    f"\n var: {np.array(posterior_predictive[ppl][i]).var()}"
                )

    # generate plots and save
    plot, plot_data = generate_plots(
        posterior_predictive=posterior_predictive,
        timing_info=timing_info,
        args_dict=args_dict,
    )
    plot.savefig("./outputs/" + timestamp + "/posterior_convergence_behaviour.png")
    # save data
    save_data(
        args_dict,
        generated_data,
        posterior_samples,
        posterior_predictive,
        timing_info,
        timestamp,
        plot_data,
    )


if __name__ == "__main__":
    main()
