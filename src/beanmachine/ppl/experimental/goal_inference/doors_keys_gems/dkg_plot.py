# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from beanmachine.facebook.goal_inference.doors_keys_gems.dkg_definitions import DKGState

from beanmachine.facebook.goal_inference.plotting.utils import (
    create_gif,
    format_gem_data,
    get_gem_figure,
    get_gem_inference_plot,
)

from matplotlib import colors


""" Set Plot Settings.
Colors are defined as follows:
Empty -> White
Wall -> Black
Door -> Grey
Key -> Gold
Agent -> Red
Gem -> blue, green, or purple
 """

CMAP = colors.ListedColormap(
    ["white", "black", "grey", "gold", "red", "blue", "green", "purple"]
)
BOUNDS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)


def plot(state: DKGState):
    """Creates graphic for current state in a Doors, Keys, and Gems problem

    Arguments:
        state: State to plot
    """
    fig, ax = get_gem_figure(state, is_doors_keys_gems=True)
    env_data, hold_data = get_data_matrix(state)

    ax[0].imshow(
        env_data,
        cmap=CMAP,
        norm=NORM,
        origin="lower",
        interpolation="none",
        extent=[0.5, state.width + 0.5, 0.5, state.height + 0.5],
    )
    ax[1].imshow(
        hold_data,
        cmap=CMAP,
        norm=NORM,
        origin="lower",
        interpolation="none",
        extent=[0.5, 1.5, 0.5, 8.5],
    )

    plt.show()
    plt.close()


def plot_gif(
    states: List[DKGState],
    inference_data: Optional[List[Dict[str, int]]] = None,
    file_path: Optional[str] = None,
) -> None:
    """Creates a gif from a list of states and inference data (if available)

    Arguments:
        states: Series of states to turn into a gif
        inference_data: Computation of the posterior P(goal|observations) at each time step
        file_path: Path to location to save the gif
    """

    fig, ax = get_gem_figure(states[0], inference_data, is_doors_keys_gems=True)

    def animation_function(frame):
        env_data, hold_data = get_data_matrix(states[frame])
        ax[0].imshow(
            env_data,
            cmap=CMAP,
            norm=NORM,
            origin="lower",
            interpolation="none",
            extent=[0.5, states[frame].width + 0.5, 0.5, states[frame].height + 0.5],
        )
        ax[1].imshow(
            hold_data,
            cmap=CMAP,
            norm=NORM,
            origin="lower",
            interpolation="none",
            extent=[0.5, 1.5, 0.5, 8.5],
        )
        if inference_data is not None:
            prob_goal = get_gem_inference_plot(inference_data, frame)
            for goal in prob_goal:
                ax[2].plot(prob_goal[goal], color=CMAP((4.0 + int(goal[3])) / 8.0))

    create_gif(fig, animation_function, states, file_path)

    plt.close()


def get_data_matrix(state: DKGState) -> Tuple[np.ndarray, np.ndarray]:
    """Computes formatted environment and holding data

    Arguments:
        state: The current state

    Returns:
        data_env: Correctly formatted data array of the environment
        data_hold: Correctly formatted data array of the held gems
    """

    data_env = get_env_data(state)

    data_env = format_gem_data(data_env)

    data_hold = get_hold_data(state)

    data_hold = format_gem_data(data_hold)

    return data_env, data_hold


def get_env_data(state: DKGState) -> np.ndarray:

    """Computes environment data matrix (defining colors) based on current environment.

    Arguments:
        state: The current state

    Returns:
        data_env: Data array describing environment (positions of agent/Items/doors/walls etc)
    """

    data_env = np.zeros((state.width, state.height))
    for i in range(1, state.width + 1):
        for j in range(1, state.height + 1):
            if (i, j) in state.walls:
                data_env[i - 1, j - 1] = 1.0
            elif (i, j) in state.doors:
                data_env[i - 1, j - 1] = 2.0

    for k in state.keys:
        if k in state.at:
            x_pos, y_pos = state.at[k]
            data_env[x_pos - 1, y_pos - 1] = 3.0

    data_env[state.x - 1, state.y - 1] = 4.0

    for g in state.gems:
        if g in state.at:
            x_pos, y_pos = state.at[g]
            # Make sure that each gem is a different color ("gem1" -> blue, "gem2" -> green, "gem3" -> purple)
            data_env[x_pos - 1, y_pos - 1] = 4.0 + int(g[3])

    return data_env


def get_hold_data(state: DKGState) -> np.ndarray:
    """Computes holding data matrix (defining colors) based on objects agent currently holds

    Arguments:
        state: The current state

    Returns:
        data_hold: Data array containing identities of the held gems
    """
    data_hold = np.zeros((1, 8))

    count = 0

    for item_id in state.keys:
        if item_id in state.has:
            data_hold[0, count] = 3.0
            count += 1

    for item_id in state.gems:
        if item_id in state.has:
            data_hold[0, count] = 4.0 + int(item_id[3])
            count += 1

    return data_hold
