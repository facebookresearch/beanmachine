# PPLBench System Specification

## Abstract

PPLBench is a benchmark for analyzing and comparing the accuracy and performance of Probabilistic Programming Languages and Libraries (PPLs) over various probabilistic models. It is designed to be open and modular such that contributors can add new models, PPL implementations, and evaluation metrics to the benchmark.
The typical PPLBench flow will be as follows:

* Establish a model with ground-truth parameter distributions; simulate train and test data from this model.
* Using the training data, train various PPL implementations; obtain posterior samples of parameters and timing information.
* Use test data, posterior samples and timing information to compare PPL implementations on various evaluation metrics.

## System Specification

### Project directory Layout:

```
`/README.md - getting started and running an example model`
`/PPLBench.py - main script`
`/models/<model name>Model.py - individual python files each containing a model module`
`/ppls/<PPL name>/<model name>.py`
`/docs/specifications.md - detailed documentation`
```

### Overview:

PPLBench will be implemented in python3. Each Model in PPLBench will have a module in ‘models/’ directory. The PPL implementations of each model will be organized in the ‘ppls/’ directory.  The ‘PPLBench’ code will consist of initializing the relevant model class based on CLI user inputs and displaying/storing output files.
[Image: system_overview.png]
### Model Module Specifications:

* Every Model should consist of module description on top. This should contain the name of the model, methods contained in the module, Model specification, and the order in which to pass model-specific arguments
* There shall be one and only one data generation method (`generate_data()`) for simulating train and test data. Method may only use numpy and scipy packages, and cannot use PPLs. The data generation should use a standard random number generator and use a fixed seed that can be user-specified through the option (`--rng-seed`). This is to ensure that the results can be reproduced.
    * Inputs: model specifications, ground-truth parameter distributions
    * Outputs: simulated train and test data, ground truth posterior predictive
* There shall be at least one ‘Evaluation Metric’ method. If only one, the metric must be the posterior predictive log likelihood of test data given posterior parameter samples (`evaluate_posterior_predictive()`). Method may only use numpy and scipy packages, cannot use PPLs
    * Inputs: posterior parameter samples, test data
    * Outputs: deliverables of the evaluation metric (e.g., posterior predictive plots, performance plots, raw sample and chain data, distance from ground truth etc.)

### PPL Implementation module specifications:

* Each PPL implementation module will contain a method `obtain_posterior()` . If PPL in not based on python, this implementation must have a python interface/wrapper to the PPL kernel. Multiple inference types are supported through the `--inference-type` option.Compilation and Sampling processes must be timed.
    * Inputs: model specification, training data, inference type
    * Outputs: posterior samples of model parameters, compile time and avg. sampling time information

### Dataflow specifications:

PPLBench uses the `args_dict` dictionary to pass various model and ppl-specific arguments. Here is the list of all the keys in this dictionary and thier brief descriptions:
 * `model`: the model for the current PPLBench run.
 * `k`: the covariate equivalent parameter of the model (e.g. k is number of covariates in the robust regression model, in noisy-or topic model it is the number of topics).
 * `n`: the number of datapoints to simulate
 * `ppls`: list of ppls that will be compared in the PPLBench run.
 * `inference_type`: type of inference (mcmc, vi).
 * `runtime`: Approximate time each ppl inference should run for.
 * `model_args`: a list of model-specific arguments; changes for each model. Refer to model descriptions for more details.
 * `rng_seed`: random seed for reproducability; default is 42.
 * `train_test_ratio`: ratio of train and test data; default is 0.5.
 * `iterations`: number of times to repeat inference; used to measure consistency across inference runs.
 * `save_samples`: option to save generated samples.
 * `save_generated_data` : option to save generated data.
 * `include_compile_time`: option to consider the compiliation time in the final plot.
 * `plot_data_size`: number of entries to save from the final plot data.
 * `num_samples_<ppl>`: number of samples to run inference for the specific ppl; this is determined automatically. 
 * `thinning_<ppl>`: amount of thinning to be applied while storing samples; detemined automatically.

Additionally, PPLBench uses the following dictionaries for storing and passing samples and timing info:
* `posterior_samples = {‘PPLs’}`
    * `PPL = {‘iterations’}`
        * `iteration = {‘samples’}`
            * `sample = {‘parameter’ : value}`
* `posterior_predictives = {‘PPLs’}`
    * `PPL = ndarray of shape [iterations, samples]`
* `timing_info = {‘PPLs’}`
    * `PPL = {‘iterations’}`
        * `iteration = {‘compile_time’, ‘inference_time’}`

### Contributing to PPLBench:

* Adding Models : Contributors can add new model module with well defined model specifications and parameters.
    * It must have a `get_defaults()` method, which takes the argument dictionary (`args_dict`) and replaces any parameters that have value `‘default’` with the default values for that model.
    * It must have a `generate_data()`method. Must accept model specs and params as well as generate train and test data in the format specified in dataflow specifications.
    * The primary metric must be average posterior predictive log likelihood of test data given posterior parameter samples (`evaluate_posterior_predictive()`). Additional metrics shall be allowed if they follow data format specified in dataflow specifications.
    * Naming Convention: capitalizeEachWordModel.py (name of Model; must end with 'Model.py')
    * See example here
* Adding PPL implementation modules: Contributors can add new PPL implementation modules in the corresponding PPL folder:
    * It must contain an `obtain_posterior()` method.
    * Naming Convention: capitalizeEachWord.py (name of model; should NOT be followed by suffix 'Model')
    * See example here

### Prerequisites:

Recommended setup:

* Linux(Tested on Ubuntu 16.04)
* Anaconda

PPLBench:

* numpy
* scipy
* torch
* pandas
* matplotlib
* argparse

PPL Implementations:
Primary:

* pyjags: see README.md for installation procedure
* pystan: (`pip/conda install pystan`)

Optional:

* pyro: (`pip/conda install pyro-ppl`) OR (install from [source](https://github.com/pyro-ppl/pyro))
* pymc3: (`pip/conda install pymc3`)
* any other PPLs supported in future

### User interface:

Arguments shall be passed through initial inputs as arguments to python script. These will have a ‘default’ option, which when passed to the models will let the model assign model-specific default values:

* `-m model`
* `-t time to run inference per PPL`
* `-n size of data to be simulated`
* `-k number of covariates`
* `-l ppls`
* `--train-test-ratio `
* `--inference-type`
* `--iterations`
* `--rng-seed`
* `--model-args model-specific arguments`
* `--plot-data-size`
