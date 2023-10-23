# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import Tuple


def check_intersection(
    obstacle_position: Tuple[float, float],
    obstacle_size: float,
    agent_position: Tuple[float, float],
    agent_size: float,
) -> bool:
    """Determines whether the agent intersects an obstacle by computing whether to two rectangles intersect

    Arguments:
        obstacle_position: Center of the obstacle
        obstacle_size: Size of the obstacle
        agent_position: Center of the agent
        agent_size: Size of the agent

    Returns:
        check_intersection: Whether the agent intersects with the object
    """
    return not (
        obstacle_position[0] + obstacle_size < agent_position[0] - agent_size
        or obstacle_position[0] - obstacle_size > agent_position[0] + agent_size
        or obstacle_position[1] + obstacle_size < agent_position[1] - agent_size
        or obstacle_position[1] - obstacle_size > agent_position[1] + agent_size
    )
