# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import Optional, Tuple, Union

from beanmachine.facebook.goal_inference.continuous_gems.cgem_definitions import (
    CGemStateNode,
)

from beanmachine.facebook.goal_inference.doors_keys_gems.dkg_definitions import (
    DKGStateNode,
)


def manhattan_distance(
    coordinates_one: Union[Tuple[float, float], Tuple[int, int]],
    coordinates_two: Union[Tuple[float, float], Tuple[int, int]],
) -> Union[int, float]:
    """Computes the manhattan distance between two objects in 2D

    Arguments:
        coordinates_one: The position of object one
        coordinates_two: The position of object two

    Returns:
        dist: The manhattan distance between the two objects

    """
    dist_x = coordinates_two[0] - coordinates_one[0]
    dist_y = coordinates_two[1] - coordinates_one[1]
    return abs(dist_x) + abs(dist_y)


def manhattan_gem_heuristic(
    curr_node: Union[DKGStateNode, CGemStateNode],
    goal: Optional[Tuple[str, str]] = None,
) -> Union[int, float]:
    """
    Defines a simple heuristic for solving gems problems with an A* solver.
    Summary: Estimates additional cost as manhattan distance to goal
             If a goal is not provided, the goal of the state is presumed

    Arguments:
        curr_node: The current node to analyze
        goal: The targeted goal

    Returns:
        manhattan_dist: Manhattan distance of node to target goal
    """
    if goal is None:
        goal = curr_node.state.goal
    if goal[1] not in curr_node.state.at:
        return 0
    final_x, final_y = curr_node.state.at[goal[1]]
    # Compute Manhattan distance to goal
    manhattan_dist = abs(final_x - curr_node.state.x) + abs(final_y - curr_node.state.y)
    return manhattan_dist
