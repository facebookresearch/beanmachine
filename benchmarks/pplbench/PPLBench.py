# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

import argparse
import csv
import datetime
import importlib
import os
import pickle
import pkgutil
import time
import traceback

import matplotlib.pyplot as plt
import models
import numpy as np
import pandas as pd
import ppls


# helper function(s)
def get_lists():
    models_list = [
        m.name[:-5]  # strip the "Model" suffix
        for m in pkgutil.iter_modules(models.__path__)
        if m.name.endswith("Model")
    ]
    ppls_list = [m.name for m in pkgutil.iter_modules(ppls.__path__)]
    return models_list, ppls_list


def logspace_datadump(averaged_pp_list, x_axis_list, x_axis_name, plot_data_size):
    """
    Create a datadump which uniformly subsamples in logspace
    Inputs:
    - averaged_pp_list: 2D array of average posterior predictive log-likelihoods
                        with shape [trials, samples]
    - x_axis_list: if x_axis_name is "time" -> list of timestamps corresponding to
                each sample in averaged_pp_list, shape [trials, samples]
                if x_axis_name is "samples" -> list of sample number (1..num_samples)
    - x_axis_name: "time" if x-axis is time, "samples" if x_axis is samples
    - plot_data_size(int): size of the subsampled arrays averaged_pp_list and time_axis

    Outputs:
    - data_dict: 'trials': logspace subsampled trials of averaged_pp_list
                 'time_axes':  logspace subsampled time axes
                 'log_indices': list of indices which are uniform in logspace.
    """
    data_dict = {}
    log_indices = []
    if x_axis_name == "time":
        end = np.log10(x_axis_list.shape[1])
    else:
        end = np.log10(len(x_axis_list))
    log_pre_index = np.logspace(0, end, num=plot_data_size, endpoint=False)
    for j in log_pre_index:
        if not int(j) in log_indices:
            log_indices.append(int(j))
    data_dict["log_indices"] = log_indices
    data_dict["trials"] = averaged_pp_list[:, log_indices]

    if x_axis_name == "time":
        data_dict["time_axes"] = x_axis_list[:, log_indices]
    else:
        data_dict["time_axes"] = np.zeros((averaged_pp_list.shape[0], len(log_indices)))

    return data_dict


def generate_plot(
    x_axis_list,
    x_axis_name,
    x_axis_min,
    x_axis_max,
    averaged_pp_list,
    args_dict,
    PPL,
    trial,
):
    """
    Helper function to generate plots of posterior log likelihood
    against either time or samples

    returns-
    plt - a matplotlib object with the plots
    plt_datadump - data for generating plots
    """
    K = args_dict["k"]
    N = args_dict["n"]
    train_test_ratio = float(args_dict["train_test_ratio"])

    # plot!
    plt.xlim(left=x_axis_min, right=x_axis_max)
    plt.grid(b=True, axis="y")
    if x_axis_name == "time":
        plt.xscale("log")
    plt.title(
        f'{args_dict["model"]} model \n'
        f"{int(int(N)*train_test_ratio)} data-points \
| {K} covariates | {trial + 1} trials"
    )
    averaged_pp_list = np.array(averaged_pp_list)
    x_axis_list = np.array(x_axis_list)
    max_line = np.max(averaged_pp_list, axis=0)
    min_line = np.min(averaged_pp_list, axis=0)
    mean_line = np.mean(averaged_pp_list, axis=0)

    if x_axis_name == "time":
        mean_x_axis_list = np.mean(x_axis_list, axis=0)
    else:
        mean_x_axis_list = x_axis_list
    label = args_dict[f"legend_name_{PPL}"]
    plt.plot(mean_x_axis_list, mean_line, label=label)
    plt.fill_between(
        mean_x_axis_list, y1=max_line, y2=min_line, interpolate=True, alpha=0.3
    )

    plt_datadump = logspace_datadump(
        averaged_pp_list, x_axis_list, x_axis_name, int(args_dict["plot_data_size"])
    )

    return plt, plt_datadump


def generate_plot_against_time(timing_info, posterior_predictive, args_dict):
    """
    Generates plots of posterior predictive log-likelihood against time

    inputs-
    timing_info(dict): 'PPL'->'compile_time' and 'inference_time'
    posterior_predictive(dict): 'PPL'->'trial'->array of posterior_predictives

    returns-
    plt - a matplotlib object with the plots
    plt_data - a dict containing data for generating plots
    """
    plt_data = {}
    for PPL in posterior_predictive.keys():
        averaged_pp_list = []
        time_axis_list = []
        for trial in range(len(posterior_predictive[PPL])):
            # timing_info[PPL][trial] contains 'inference_time'
            # and 'compile_time' for each trial of each PPL
            sample_time = timing_info[PPL][trial]["inference_time"]
            # posterior_predictive[PPL][trial] is a 1D array of
            # posterior_predictives for a certain PPL for certain trial
            averaged_pp = np.zeros_like(posterior_predictive[PPL][trial])
            time_axis = np.linspace(
                start=0, stop=sample_time, num=len(posterior_predictive[PPL][trial])
            )
            if args_dict["include_compile_time"] == "yes":
                time_axis += timing_info[PPL][trial]["compile_time"]
            for i in range(len(posterior_predictive[PPL][trial])):
                averaged_pp[i] = np.mean(posterior_predictive[PPL][trial][: i + 1])
            averaged_pp_list.append(averaged_pp)
            time_axis_list.append(time_axis)

        plt, plt_datadump = generate_plot(
            time_axis_list,
            "time",
            0.01,
            time_axis[-1],
            averaged_pp_list,
            args_dict,
            PPL,
            trial,
        )
        plt_data[PPL] = plt_datadump
    plt.legend()
    plt.ylabel("Average log predictive")
    plt.xlabel(
        "Time(Seconds)",
        "NOTE: includes compile time"
        if args_dict["include_compile_time"] == "yes"
        else None,
    )

    return plt, plt_data


def generate_plot_against_sample(posterior_predictive, args_dict):
    """
    Generates plots of posterior predictive log-likelihood against number of samples

    inputs-
    posterior_predictive(dict): 'PPL'->'trial'->array of posterior_predictives

    returns-
    plt - a matplotlib object with the plots
    plt_data - a dict containing data for generating plots
    """
    plt_data = {}
    for PPL in posterior_predictive.keys():
        sample_axis_list = [
            i + 1
            for i in range(
                int(args_dict[f"num_samples_{PPL}"] / args_dict[f"thinning_{PPL}"])
            )
        ]
        averaged_pp_list = []

        for trial in range(len(posterior_predictive[PPL])):
            # posterior_predictive[PPL][trial] is a 1D array of
            # posterior_predictives for a certain PPL for certain trial
            averaged_pp = np.zeros_like(posterior_predictive[PPL][trial])
            for i in range(len(posterior_predictive[PPL][trial])):
                averaged_pp[i] = np.mean(posterior_predictive[PPL][trial][: i + 1])
            averaged_pp_list.append(averaged_pp)

        plt, plt_datadump = generate_plot(
            sample_axis_list,
            "samples",
            1,
            sample_axis_list[-1],
            averaged_pp_list,
            args_dict,
            PPL,
            trial,
        )
        plt_data[PPL] = plt_datadump
    plt.legend()
    plt.ylabel("Average log predictive")
    plt.xlabel("Samples")

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
    timing_info,
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
    plot_data: data structure that stores information to recreate the plots
    """
    x_axis_names = ["samples", "time"]
    for i in range(len(plot_data)):
        with open(
            os.path.join(
                args_dict["output_dir"], "plot_data_{}.csv".format(x_axis_names[i])
            ),
            "w",
        ) as csv_file:
            csvwriter = csv.writer(csv_file, delimiter=",")
            csvwriter.writerow(
                ["ppl", "trial", "sample", "average_log_predictive", "time"]
            )
            for ppl in plot_data[i]:
                for trial in range(plot_data[i][ppl]["trials"].shape[0]):
                    for sample in range(len(plot_data[i][ppl]["trials"][trial, :])):
                        csvwriter.writerow(
                            [
                                ppl,
                                trial + 1,
                                plot_data[i][ppl]["log_indices"][sample]
                                * args_dict[f"thinning_{ppl}"],
                                plot_data[i][ppl]["trials"][trial, sample],
                                plot_data[i][ppl]["time_axes"][trial, sample],
                            ]
                        )

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

    with open(
        os.path.join(args_dict["output_dir"], "timing_info.csv"), "w"
    ) as csv_file:
        pd.DataFrame.from_dict(timing_info, orient="index").to_csv(csv_file)

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
            posterior_samples[ppl][i], timing_info[ppl][i] = module.obtain_posterior(
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
                f"Trial {i + 1} "
                f'complete in {timing_info[ppl][i]["inference_time"]} '
                "seconds.\n Statistics of  posterior predictive\n mean:"
                f"{np.array(posterior_predictive[ppl][i]).mean()}"
                f"\n var: {np.array(posterior_predictive[ppl][i]).var()}"
            )

    # generate plots and save
    plot_time, plot_data_time = generate_plot_against_time(
        timing_info=timing_info,
        posterior_predictive=posterior_predictive,
        args_dict=args_dict,
    )

    plot_time.savefig(
        os.path.join(
            args_dict["output_dir"], "time_posterior_convergence_behaviour.png"
        )
    )
    plot_time.clf()
    plot_sample, plot_data_sample = generate_plot_against_sample(
        posterior_predictive=posterior_predictive, args_dict=args_dict
    )
    plot_sample.savefig(
        os.path.join(
            args_dict["output_dir"], "sample_posterior_convergence_behaviour.png"
        )
    )
    plot_data = [plot_data_sample] + [plot_data_time]

    # save data
    save_data(
        args_dict,
        generated_data,
        posterior_samples,
        posterior_predictive,
        timing_info,
        plot_data,
    )
    print(f"Output in : {args_dict['output_dir']}")


if __name__ == "__main__":
    main()
