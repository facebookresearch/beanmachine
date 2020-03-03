from typing import Dict, Tuple

import benchmarks.pplbench.models.seismicLocationUtil as seismic
import numpy as np
import torch
import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.inference.proposer.single_site_hamiltonian_monte_carlo_proposer import (
    SingleSiteHamiltonianMonteCarloProposer,
)
from beanmachine.ppl.model.utils import RVIdentifier
from beanmachine.ppl.world import World
from benchmarks.pplbench.ppls.beanmachine.seismicProjectionModel import (
    SeismicProjectionModel,
)
from torch import Tensor


class SingleSiteSeismicProposer(SingleSiteAncestralProposer):
    def __init__(self):
        super().__init__()
        self.time_models = {}
        self.x_models = {}
        self.y_models = {}
        self.detected_stations = {}
        self.init_models = False
        self.init_detected = False

    def _init_detected_stations(self, node: RVIdentifier, world: World):
        node_var = world.get_node_in_world_raise_error(node, False)
        for child in node_var.children:
            if (
                child is None
                or child.function._wrapper != SeismicProjectionModel.is_detected
            ):
                continue
            child_var = world.get_node_in_world_raise_error(child, False)
            self.detected_stations = torch.nonzero(child_var.value).squeeze()
        self.init_detected = True

    def _compute_models(self, node: RVIdentifier, world: World):
        self.time_models = {}
        self.x_models = {}
        self.y_models = {}
        node_var = world.get_node_in_world_raise_error(node, False)

        for child in node_var.children:
            if (
                child is None
                or child.function._wrapper != SeismicProjectionModel.detection
            ):
                continue
            child_var = world.get_node_in_world_raise_error(child, False)

            for i in self.detected_stations:
                station = int(i)

                time = child_var.value[i][0]
                azi = child_var.value[i][1]
                slow = child_var.value[i][2]

                distance = seismic.invert_slowness(slow)
                lon, lat = seismic.invert_dist_azimuth(station, distance, azi)
                lon = torch.fmod(180 + lon, 360).sub_(180)
                lat = torch.fmod(180 + lat, 360).sub_(180)
                ttime = seismic.compute_travel_time(distance)
                event_time = time - ttime

                x, y = seismic.convert_lon_lat_to_projection(lon, lat)
                scale_x = (
                    seismic.convert_lon_lat_to_projection(lon, lat)[0]
                    - seismic.convert_lon_lat_to_projection(lon.add_(5), lat.add_(5))[0]
                )
                scale_y = (
                    seismic.convert_lon_lat_to_projection(lon, lat)[1]
                    - seismic.convert_lon_lat_to_projection(lon.add_(5), lat.add_(5))[1]
                )

                self.time_models[station] = dist.Normal(event_time, 20.0)
                self.x_models[station] = dist.Normal(x, torch.abs(scale_x))
                self.y_models[station] = dist.Normal(y, torch.abs(scale_y))
        self.init_models = True

    def _propose_probability(self, node, proposal, num_detections):
        time_prob = []
        x_prob = []
        y_prob = []
        for station in self.time_models:
            time_prob.append(self.time_models[station].log_prob(proposal[0]).sum())
            x_prob.append(self.x_models[station].log_prob(proposal[1]).sum())
            y_prob.append(self.y_models[station].log_prob(proposal[2]).sum())

        model_prob = (
            torch.logsumexp(tensor(time_prob), 0)
            + torch.logsumexp(tensor(x_prob), 0)
            + torch.logsumexp(tensor(y_prob), 0)
        )
        model_prob = model_prob - np.log(num_detections) + np.log(0.9)

        uniform_prob = 4 / np.pi / (
            (4 + proposal[1] ** 2 + proposal[2] ** 2) ** 2
        ) + np.log(0.1)

        return torch.logsumexp(torch.stack((model_prob, uniform_prob)), 0)

    def propose(self, node: RVIdentifier, world: World) -> Tuple[Tensor, Tensor, Dict]:
        """
        Proposes a new value for the event node.

        :param node: the node for which we'll need to propose a new value for.
        :param world: the world in which we'll propose a new value for node.
        :returns: a new proposed value for the node, which either uses an HMC proposer
        or a mixture model combining the station-prediction GMM with a uniform proposer
        """
        use_hmc = dist.Bernoulli(0.5).sample()

        if not self.init_detected:
            self._init_detected_stations(node, world)
        if not self.init_models:
            self._compute_models(node, world)

        node_var = world.get_node_in_world_raise_error(node, False)
        num_detections = len(self.detected_stations)

        if use_hmc or num_detections == 0:
            hmc = SingleSiteHamiltonianMonteCarloProposer(
                tensor([1.0, 0.002, 0.002]), 10
            )
            return hmc.propose(node, world)
        else:
            use_uniform = dist.Bernoulli(0.1).sample()
            if use_uniform:
                proposal, _, _ = super().propose(node, world)
            else:
                time_index = dist.Categorical(
                    torch.ones(num_detections) / (num_detections)
                ).sample()
                time_model = self.time_models[int(self.detected_stations[time_index])]
                x_index = dist.Categorical(
                    torch.ones(num_detections) / (num_detections)
                ).sample()
                x_model = self.x_models[int(self.detected_stations[x_index])]
                y_index = dist.Categorical(
                    torch.ones(num_detections) / (num_detections)
                ).sample()
                y_model = self.y_models[int(self.detected_stations[y_index])]
                proposal = torch.stack(
                    (time_model.sample(), x_model.sample(), y_model.sample())
                )

            return (
                proposal,
                self._propose_probability(node, node_var.value, num_detections)
                - self._propose_probability(node, proposal, num_detections),
                {},
            )

    def post_process(
        self, node: RVIdentifier, world: World, auxiliary_variables: Dict
    ) -> Tensor:
        """
        Computes the log probability of going back to the old value.

        :param node: the node for which we'll need to propose a new value for.
        :param world: the world in which we have already proposed a new value
        for node.
        :returns: the log probability of proposing the old value from this new world.
        """
        return tensor(0.0)
