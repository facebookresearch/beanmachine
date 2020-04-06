import time
from typing import Dict, List, Tuple

import torch.tensor as tensor
from beanmachine.ppl.inference.compositional_infer import CompositionalInference
from benchmarks.pplbench.ppls.beanmachine.seismic_projection_model import (
    SeismicProjectionModel,
)
from benchmarks.pplbench.ppls.beanmachine.seismic_proposer import (
    SingleSiteSeismicProposer,
)


def obtain_posterior(data_train: Dict, args_dict: Dict, model) -> Tuple[List, Dict]:
    """
    Obtain samples using beanmachine

    :param args_dict: arguments dictionary
    :param model: dictionary with
        mu(int) = detection probability properties per station
        theta_k(int) = detection properties per station
    :returns: generated_data(dict) = {
        data_train: {
            is_detected: is_detected for stations 1-9,
            detection: detection for stations 1-9
        data_test: {
            detection_zero: detection for station 0
        }
    """
    model_is_detected = data_train["is_detected"]
    model_detection = data_train["detection"]
    mu = model["mu"]
    theta_k = model["theta_k"]

    start_time = time.time()
    model = SeismicProjectionModel()
    mh = CompositionalInference({model.event: SingleSiteSeismicProposer()})
    elapsed_time_compile_beanmachine = time.time() - start_time

    obs = {
        model.holdout_is_detected(): tensor(1.0),
        model.is_detected(): model_is_detected,
        model.detection(): model_detection,
        model.theta_k(): theta_k,
        model.mu_k_d(): mu,
    }

    query = [model.event(), model.event_magnitude()]
    num_samples = args_dict["num_samples_beanmachine"]

    start_time = time.time()
    mh_infer = mh.infer(query, obs, num_samples, 1)
    elapsed_time_sample_beanmachine = time.time() - start_time

    mag = mh_infer[model.event_magnitude()].squeeze().detach()
    event = mh_infer[model.event()].squeeze().detach()

    samples = []
    for i in range(num_samples):
        samples.append((event[i], mag[i]))

    timing_info = {
        "compile_time": elapsed_time_compile_beanmachine,
        "inference_time": elapsed_time_sample_beanmachine,
    }
    return (samples, timing_info)
