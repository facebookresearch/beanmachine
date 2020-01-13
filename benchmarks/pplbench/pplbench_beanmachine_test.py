# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import benchmarks.pplbench.models.hiddenMarkovModel as hmm
import benchmarks.pplbench.models.robustRegressionModel as robustregression_model
import benchmarks.pplbench.ppls.beanmachine.hiddenMarkov as bm_hmm
import benchmarks.pplbench.ppls.beanmachine.robustRegression as bm_robustregression


# import benchmarks.pplbench.ppls.pymc3.hiddenMarkov as pm_hmm
# from PPLBench import get_args


class PPLBenchBeanMachineTest(unittest.TestCase):
    # def setUp(self):

    # Re-implemented skeleton of PPLBench.main, far fewer lines of code
    def run_execution(self, model, ppl_module):

        # Get args_dict.
        # args = get_args([], [])
        # args_dict = vars(args)
        args_dict = model.get_defaults()
        args_dict.update(
            {
                "n": 10,
                "k": 10,
                "trials": 1,
                "num_samples": 5,
                "num_samples_beanmachine": 5,
                "rng_seed": 11,
            }
        )

        # Generate model and data.
        model_instance = model.generate_model(args_dict)
        generated_data = model.generate_data(args_dict=args_dict, model=model_instance)

        # Run inference and compute PPC likelihoods.
        samples, time_info = ppl_module.obtain_posterior(
            generated_data["data_train"], args_dict, model_instance
        )
        pred_log_lik_array = model.evaluate_posterior_predictive(
            samples, generated_data["data_test"], model_instance
        )

        # Check posterior samples.
        self.assertEqual(len(pred_log_lik_array), args_dict["num_samples"])
        self.assertGreater(len(pred_log_lik_array), 0)
        self.assertTrue(pred_log_lik_array[0])

    def test_beanmachine_hmm(self):
        self.run_execution(hmm, bm_hmm)

    def test_beanmachine_robustregression(self):
        self.run_execution(robustregression_model, bm_robustregression)

    # This cannot be implemented, as Buck does not have access to PyMC3.
    # def test_pymc3(self):
    # self.run_execution(pm_lib)
