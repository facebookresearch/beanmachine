from typing import Tuple

import numpy as np
import torch
import torch.distributions as dist
import torch.tensor as tensor
from torch import Tensor
from torch.distributions.distribution import Distribution


# helper functions
stations = tensor(
    [
        [133.9, -23.7],
        [98.9, 18.5],
        [26.1, 61.4],
        [-146.9, 64.8],
        [82.3, 46.8],
        [106.4, 47.8],
        [141.6, -31.9],
        [1.7, 13.1],
        [134.3, -19.9],
        [84.8, 53.9],
    ]
)


def convert_degrees_to_radians(deg: Tensor) -> Tensor:
    return torch.mul(deg, np.pi) / 180


def convert_radians_to_degrees(rad: Tensor) -> Tensor:
    return torch.div(rad, np.pi) * 180


def convert_lon_lat_to_cartesian(
    lon: Tensor, lat: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    lon = convert_degrees_to_radians(lon)
    lat = convert_degrees_to_radians(lat)
    x = torch.cos(lon) * torch.cos(lat)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    return x, y, z


def convert_cartesian_to_lon_lat(
    x: Tensor, y: Tensor, z: Tensor
) -> Tuple[Tensor, Tensor]:
    lon = convert_radians_to_degrees(torch.atan2(y, x))
    lat = convert_radians_to_degrees(torch.asin(z))
    return lon, lat


def convert_cartesian_to_projection(
    x: Tensor, y: Tensor, z: Tensor
) -> Tuple[Tensor, Tensor]:
    return (x / (1 - z), y / (1 - z))


def convert_projection_to_cartesian(
    a: Tensor, b: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    frac = 1 / (1 + a ** 2 + b ** 2)
    x = (2 * a) * frac
    y = (2 * b) * frac
    z = (-1 + a ** 2 + b ** 2) * frac
    return x, y, z


def convert_projection_to_lon_lat(a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    x, y, z = convert_projection_to_cartesian(a, b)
    return convert_cartesian_to_lon_lat(x, y, z)


def convert_lon_lat_to_projection(lon: Tensor, lat: Tensor) -> Tuple[Tensor, Tensor]:
    x, y, z = convert_lon_lat_to_cartesian(lon, lat)
    return convert_cartesian_to_projection(x, y, z)


def compute_delta(lon1: Tensor, lat1: Tensor, lon2: Tensor, lat2: Tensor) -> Tensor:
    lng1, lat1 = convert_degrees_to_radians(lon1), convert_degrees_to_radians(lat1)
    lng2, lat2 = convert_degrees_to_radians(lon2), convert_degrees_to_radians(lat2)

    sin_lat1, cos_lat1 = torch.sin(lat1), torch.cos(lat1)
    sin_lat2, cos_lat2 = torch.sin(lat2), torch.cos(lat2)

    delta_lng = lng2 - lng1
    cos_delta_lng, sin_delta_lng = torch.cos(delta_lng), torch.sin(delta_lng)

    angle = torch.atan2(
        torch.sqrt(
            (cos_lat2 * sin_delta_lng) ** 2
            + (cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_delta_lng) ** 2
        ),
        sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng,
    )

    return convert_radians_to_degrees(angle)


def compute_azimuth(lon1: Tensor, lat1: Tensor, lon2: Tensor, lat2: Tensor) -> Tensor:
    lng1, lat1 = convert_degrees_to_radians(lon1), convert_degrees_to_radians(lat1)
    lng2, lat2 = convert_degrees_to_radians(lon2), convert_degrees_to_radians(lat2)

    delta_lon = lng2 - lng1

    y = torch.sin(delta_lon)

    x = torch.cos(lat1) * torch.tan(lat2) - torch.sin(lat1) * torch.cos(delta_lon)

    azi = convert_radians_to_degrees(torch.atan2(y, x))

    return (azi + 360) % 360


def compute_great_circle_distance(
    lon1: Tensor, lat1: Tensor, lon2: Tensor, lat2: Tensor
) -> Tensor:
    lng1, lat1 = convert_degrees_to_radians(lon1), convert_degrees_to_radians(lat1)
    lng2, lat2 = convert_degrees_to_radians(lon2), convert_degrees_to_radians(lat2)

    delta_lon = lng2 - lng1

    y = torch.sin(delta_lon)

    x = torch.cos(lat1) * torch.tan(lat2) - torch.sin(lat1) * torch.cos(delta_lon)

    azi = convert_radians_to_degrees(torch.atan2(y, x))

    return (azi + 360) % 360


def compute_travel_time(delta: Tensor) -> Tensor:
    return -0.023 * delta.pow(2) + 10.7 * delta + 5


def compute_slowness(delta: Tensor) -> Tensor:
    return -0.046 * delta + 10.7


def invert_slowness(slow: Tensor) -> Tensor:
    return torch.abs((slow - 10.7) / -0.046)


def invert_dist_azimuth(
    station: int, distance: Tensor, azi: Tensor
) -> Tuple[Tensor, Tensor]:
    lon1, lat1 = stations[station]
    lon1, lat1 = convert_degrees_to_radians(lon1), convert_degrees_to_radians(lat1)
    distance, azi = (
        convert_degrees_to_radians(distance),
        convert_degrees_to_radians(azi),
    )

    lat2 = torch.asin(
        torch.sin(lat1) * torch.cos(distance)
        + torch.cos(lat1) * torch.sin(distance) * torch.cos(azi)
    )

    lon2 = lon1 + torch.atan2(
        torch.sin(azi) * torch.sin(distance) * torch.cos(lat1),
        torch.cos(distance) - torch.sin(lat1) * torch.sin(lat2),
    )

    return convert_radians_to_degrees(lon2), convert_radians_to_degrees(lat2)


def logistic(param: Tensor) -> Tensor:
    return 1 / (1 + torch.exp(-param))


class InverseGamma(Distribution):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.dist = dist.TransformedDistribution(
            dist.Gamma(alpha, beta), [dist.PowerTransform(-1)]
        )

    def mean(self):
        return self.beta / (self.alpha - 1)

    def sample(self):
        s = self.dist.sample()
        return s

    @property
    def support(self):
        return dist.constraints.greater_than(0.0)

    def log_prob(self, value):
        return self.dist.log_prob(value)


class FiniteExponential(Distribution):
    def __init__(self, scale, min, max):
        self.scale = scale
        self.min = min
        self.max = max
        self.dist = dist.Exponential(1 / scale)

    def mean(self):
        return self.scale * (1 - torch.exp(-(self.max - self.min) / self.scale))

    def sample(self):
        s = self.dist.sample() + self.min
        while s > self.max:
            s = self.dist.sample() + self.min
        return s

    @property
    def support(self):
        return dist.constraints._Real

    def log_prob(self, value):
        scale_constant = np.log(self.scale)
        constant = -np.log(1 - np.exp(-(self.max - self.min) / self.scale))
        return -scale_constant + constant - ((value - self.min) / self.scale)


class StereographicProjectionUniform(Distribution):
    def initialize(self):
        time = 1800
        lon = 0.1
        lat = 0.1
        a, b = convert_lon_lat_to_projection(lon, lat)
        return tensor([time, a, b])

    def sample(self):
        time = dist.Uniform(0, 3600).sample()
        lon = dist.Uniform(-180, 180).sample()
        lat = np.degrees(np.arcsin(dist.Uniform(-1, 1).sample()))
        a, b = convert_lon_lat_to_projection(lon, lat)
        return tensor([time, a, b])

    def log_prob(self, value):
        x = value[1]
        y = value[2]
        return 4 / np.pi / ((4 + x ** 2 + y ** 2) ** 2)

    @property
    def support(self):
        return dist.constraints._Real
