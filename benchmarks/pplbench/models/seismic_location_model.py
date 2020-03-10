# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Module for Seismic 2D Location Model

Module contains following methods:
get_defaults()
generate_model()
generate_data()
evaluate_posterior_predictive()

Model Specification:
The Seismic2D model has a seismic event on the surface
of the earth, with a specific time, lon, lat, and mag.
Different stations on the earth have a probability
of the detecting the event based on the location and
magnitude of the event. If the station detects the event,
it will detect the time, azimuth, and slowness.

Random variables:
event: time, and stereographic projection of lon and lat
event_magnitude: magnitude of event
detection_zero: the detection of the event at station 0
   (station 0 always detects the event)
   the detection consists of the time, azimuth, and slowness
is_detected: if the event was detected at stations 1-9
detection: the detection of the event at stations 1-9
    each detection has its own time, azimuth, and slowness
theta: station-specific detection parameter
    loc of Laplace for detection time, azimuth, and slowness
mu: station-specific detection parameter
    mu[0] + mu[1] * event_magnitude + mu[2] * distance
    in a logistic regression is the probability
    that a station will detected the event

Evaluation:
We calculate the predictive log_likelihood for
    station 0 detecting the detection
    the time, azimuth, and slowness at station 0.
This is calculated conditioned on
    station 0 detecting the event
    at least two other stations detecting the event
    the detections of stations 1-9 (if detected)

Model specific arguments:
return dictionary with mu and theta
"""

import benchmarks.pplbench.models.seismic_location_util as seismic
import numpy as np
import torch
import torch.distributions as dist
import torch.tensor as tensor
from tqdm import tqdm


def get_defaults():
    defaults = {
        "n": 2000,
        "k": 10,
        "runtime": 200,
        "model_args": 1,
        "train_test_ratio": 0.5,
    }
    return defaults


def generate_model(args_dict):
    """
    Generate parameters for seismic model.

    :param args_dict: arguments dictionary
    :returns: dictionary with mu and theta_k for seismic model
    """
    mean = tensor([-10.4, 3.26, -0.0499])
    cov = tensor(
        [
            [13.43, -2.36, -0.0122],
            [-2.36, 0.452, 0.000112],
            [-0.0122, 0.000112, 0.000125],
        ]
    )
    mean = mean.unsqueeze(0).expand(10, 3)
    cov = cov.unsqueeze(0).expand(10, 3, 3)
    mu = dist.MultivariateNormal(mean, cov).sample()

    alpha = tensor([120, 5.2, 6.7]).unsqueeze(0).expand(10, 3)
    beta = tensor([118, 44, 7.5]).unsqueeze(0).expand(10, 3)
    theta_k = seismic.InverseGamma(alpha, beta).sample()
    return {"mu": mu, "theta_k": theta_k}


def generate_data(args_dict, model):
    """
    Generate data for seismic location model.

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
    print("Generating data")

    torch.manual_seed(args_dict["rng_seed"])

    mu = model["mu"]
    theta_k = model["theta_k"]

    detected_at_zero = False
    multiple_detections = False
    while not detected_at_zero or not multiple_detections:
        event_time = dist.Uniform(0, 3600).sample()
        event_lon = dist.Uniform(-180, 180).sample()
        event_lat = np.degrees(np.arcsin(dist.Uniform(-1, 1).sample()))

        magnitude = seismic.FiniteExponential(3.0, 4.0, 6.0).sample()

        stations_lon, stations_lat = seismic.stations[:, 0], seismic.stations[:, 1]
        delta = seismic.compute_delta(stations_lon, stations_lat, event_lon, event_lat)
        detection_prob = seismic.logistic(
            mu[:, 0] + mu[:, 1] * magnitude + mu[:, 2] * delta
        )
        detected_at_zero = dist.Bernoulli(detection_prob[0]).sample()

        is_detected = dist.Bernoulli(detection_prob[1:]).sample()
        multiple_detections = is_detected.sum() >= 2.0

    time_loc = event_time + seismic.compute_travel_time(delta)
    azimuth_loc = seismic.compute_azimuth(
        stations_lon, stations_lat, event_lon, event_lat
    )
    slowness_loc = seismic.compute_slowness(delta)
    detection_loc = torch.stack(
        (
            time_loc,
            azimuth_loc.to(dtype=time_loc.dtype),
            slowness_loc.to(dtype=time_loc.dtype),
        ),
        dim=1,
    )

    mask = is_detected.expand(3, 9).transpose(0, 1)
    loc = detection_loc[1:] * mask
    scale = 1.0 + (theta_k[1:] - 1.0) * mask
    detection = dist.Laplace(loc, scale).sample()

    detection_zero = dist.Laplace(detection_loc[0], theta_k[0]).sample()

    data_train = {"is_detected": is_detected, "detection": detection}
    data_test = {"detection_zero": detection_zero}
    return {"data_train": data_train, "data_test": data_test}


def evaluate_posterior_predictive(samples, data_test, model):
    """
    Computes the likelihood of held-out testset wrt parameter samples

    :param samples: dictionary with parameter samples from model
    :param data_test: test data
    :param model: dictionary with
        mu(int) = detection probability properties per station
        theta_k(int) = detection properties per station
    :returns: log-likelihoods of data wrt parameter samples
    """
    detection_zero = data_test["detection_zero"]
    pred_llh = []
    theta_k = model["theta_k"]
    mu = model["mu"]

    for sample in tqdm(samples, desc="eval", leave=False):
        event, event_mag = sample
        llh = _calculate_llh(detection_zero, event, event_mag, theta_k, mu).detach()
        pred_llh.append(llh)
    # return as a numpy array of sum over test data
    return np.array(pred_llh)


def _calculate_llh(detection, event, event_mag, theta_k, mu):
    magnitude = event_mag

    loc_x = event[1]
    loc_y = event[2]
    event_lon, event_lat = seismic.convert_projection_to_lon_lat(loc_x, loc_y)
    station_lon, station_lat = seismic.stations[0][0], seismic.stations[0][1]
    delta = seismic.compute_delta(station_lon, station_lat, event_lon, event_lat)

    detection_prob = seismic.logistic(
        mu[0][0] + mu[0][1] * magnitude + mu[0][2] * delta
    )

    time_loc = event[0] + seismic.compute_travel_time(delta)
    azimuth_loc = seismic.compute_azimuth(
        station_lon, station_lat, event_lon, event_lat
    )
    slowness_loc = seismic.compute_slowness(delta)
    detection_loc = torch.stack(
        (
            time_loc,
            azimuth_loc.to(dtype=time_loc.dtype),
            slowness_loc.to(dtype=time_loc.dtype),
        )
    )
    detection_scale = theta_k[0]
    log_prob = dist.Laplace(detection_loc, detection_scale).log_prob(detection).sum()

    return log_prob + dist.Bernoulli(detection_prob).log_prob(tensor(1.0)).sum()
