# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import json

from beanmachine.facebook.goal_inference.continuous_gems.cgem_definitions import (
    CGemState,
    Gem,
)


def parse(file_path: str) -> CGemState:
    """Reads in selected continuous gems problem. Intializes state by parsing file for objects, rules, and goal.

    Arguments:
        file_path: Path pointing to the selected problem

    Returns:
        initial_state: The initial state of the continuous gems problem
    """
    with open(file_path) as problem_file:
        cgem_problem_data = json.load(problem_file)

        at = {}
        gems = {}
        for gem_id in cgem_problem_data["gem_locations"]:
            gems[gem_id] = Gem(gem_id)
            at[gem_id] = tuple(cgem_problem_data["gem_locations"][gem_id])
        has = {}
        obstacles = [tuple(obstacle) for obstacle in cgem_problem_data["obstacles"]]

    return CGemState(
        tuple(cgem_problem_data["goal"]),
        cgem_problem_data["width"],
        cgem_problem_data["height"],
        gems,
        cgem_problem_data["gem_size"],
        has,
        at,
        set(obstacles),
        cgem_problem_data["obstacle_size"],
        cgem_problem_data["agent_size"],
        cgem_problem_data["x"],
        cgem_problem_data["y"],
        cgem_problem_data["angle"],
    )
