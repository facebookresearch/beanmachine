import benchmarks.pplbench.models.seismicLocationUtil as seismic
import torch
import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.model.statistical_model import sample
from torch import Tensor


class SeismicProjectionModel(object):
    def __init__(self):
        self.T = 3600.0
        self.mu_k_t = 0
        self.mu_k_z = 0
        self.mu_k_s = 0

    @sample
    def event(self):
        return seismic.StereographicProjectionUniform()

    @sample
    def event_magnitude(self):
        return seismic.FiniteExponential(3.0, 4.0, 6.0)

    @sample
    def theta_k(self):
        alpha = tensor([120, 5.2, 6.7]).unsqueeze(0).expand(10, 3)
        beta = tensor([118, 44, 7.5]).unsqueeze(0).expand(10, 3)
        return seismic.InverseGamma(alpha, beta)

    @sample
    def mu_k_d(self):
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
        return dist.MultivariateNormal(mean, cov)

    @sample
    def is_detected(self):
        mu = self.mu_k_d()

        magnitude = self.event_magnitude()

        loc_x = self.event()[1]
        loc_y = self.event()[2]
        event_lon, event_lat = seismic.convert_projection_to_lon_lat(loc_x, loc_y)
        stations_lon, stations_lat = seismic.stations[:, 0], seismic.stations[:, 1]
        delta = seismic.compute_delta(stations_lon, stations_lat, event_lon, event_lat)

        detection_prob = seismic.logistic(
            mu[:, 0] + mu[:, 1] * magnitude + mu[:, 2] * delta
        )
        return dist.Bernoulli(detection_prob[1:])

    @sample
    def holdout_is_detected(self):
        mu = self.mu_k_d()

        magnitude = self.event_magnitude()

        loc_x = self.event()[1]
        loc_y = self.event()[2]
        event_lon, event_lat = seismic.convert_projection_to_lon_lat(loc_x, loc_y)
        stations_lon, stations_lat = seismic.stations[0][0], seismic.stations[0][1]
        delta = seismic.compute_delta(stations_lon, stations_lat, event_lon, event_lat)

        detection_prob = seismic.logistic(
            mu[0][0] + mu[0][1] * magnitude + mu[0][2] * delta
        )
        return dist.Bernoulli(detection_prob)

    def compute_detection_loc(self, event) -> Tensor:
        event_time, event_x, event_y = event[0], event[1], event[2]
        event_lon, event_lat = seismic.convert_projection_to_lon_lat(event_x, event_y)
        stations_lon, stations_lat = seismic.stations[:, 0], seismic.stations[:, 1]

        delta = seismic.compute_delta(stations_lon, stations_lat, event_lon, event_lat)

        time_loc = event_time + seismic.compute_travel_time(delta) + self.mu_k_t
        azimuth_loc = seismic.compute_azimuth(
            stations_lon, stations_lat, event_lon, event_lat
        )
        slowness_loc = seismic.compute_slowness(delta) + self.mu_k_s

        return torch.stack(
            [
                time_loc,
                azimuth_loc.to(dtype=time_loc.dtype),
                slowness_loc.to(dtype=time_loc.dtype),
            ],
            dim=1,
        )

    @sample
    def detection(self):
        mask = self.is_detected().expand(3, 9).transpose(0, 1)
        detection_loc = self.compute_detection_loc(self.event())

        loc = detection_loc[1:] * mask
        scale = 1.0 + (self.theta_k()[1:] - 1.0) * mask

        return dist.Laplace(loc, scale)
