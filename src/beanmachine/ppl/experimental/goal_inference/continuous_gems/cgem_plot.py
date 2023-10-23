# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import Dict, List, Optional

import matplotlib

import matplotlib.pyplot as plt
import numpy as np

from beanmachine.facebook.goal_inference.continuous_gems.cgem_definitions import (
    CGemState,
)

from beanmachine.facebook.goal_inference.plotting.utils import (
    create_gif,
    format_gem_data,
    get_gem_figure,
    get_gem_inference_plot,
)

from matplotlib import colors
from matplotlib.patches import Rectangle


""" Set Plot Settings.
Colors are defined as follows:
Empty -> White
Obstacle -> Black
Agent -> Red
Gem -> blue, green, or purple
"""

CMAP = colors.ListedColormap(["white", "black", "red", "blue", "green", "purple"])
BOUNDS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 2.50


def plot(state: CGemState):

    """Creates graphic for current state

    Arguments:
        state: State to plot
    """

    fig, ax = get_gem_figure(state)
    hold_data = get_data_matrix(state)
    add_data_to_axes(ax, state, hold_data)
    plt.show()
    plt.close()


def plot_gif(
    states: List[CGemState],
    inference_data: Optional[List[Dict[str, int]]] = None,
    file_path: Optional[str] = None,
) -> None:
    """Creates a gif from a list of states and inference data (if available)

    Arguments:
        states: Series of states to turn into a gif
        inference_data: Computation of the posterior P(goal|observations) at each time step
        file_path: Path to location to save the gif
    """

    fig, ax = get_gem_figure(states[0], inference_data)

    def animation_function(frame):
        state = states[frame]
        hold_data = get_data_matrix(state)
        add_data_to_axes(ax, state, hold_data)
        if inference_data is not None:
            prob_goal = get_gem_inference_plot(inference_data, frame)
            for goal in prob_goal:
                ax[2].plot(prob_goal[goal], color=CMAP(NORM(2.5 + float(goal[3]))))

    create_gif(fig, animation_function, states, file_path)

    plt.close()


def get_data_matrix(state: CGemState) -> np.ndarray:
    """Computes formatted holding data

    Arguments:
        state: The current state

    Returns:
        data_hold: Correctly formatted data array of the held gems
    """

    data_hold = get_hold_data(state)

    data_hold = format_gem_data(
        data_hold,
    )

    return data_hold


def get_hold_data(state: CGemState) -> np.ndarray:
    """Computes holding data matrix (defining colors) based on objects agent currently holds

    Arguments:
        state: The current state

    Returns:
        data_hold: Data array containing identities of the held gems
    """

    data_hold = np.zeros((1, 8))

    count = 0

    for item_id in state.gems:
        if item_id in state.has:
            data_hold[0, count] = 2.0 + int(item_id[3])
            count += 1

    return data_hold


def add_data_to_axes(
    ax: List[matplotlib.axes.Axes], state: CGemState, hold_data: np.ndarray
) -> None:
    """Adds Environment and held data to subplots

    Arguments:
        ax: List of axes objects in the current figure
        state: Current state to plot
        hold_data: Correctly formmatted data array for held gems

    """
    # Plot Agent
    ax[0].add_patch(
        Rectangle(
            (state.x - state.agent_size, state.y - state.agent_size),
            state.agent_size * 2.0,
            state.agent_size * 2.0,
            color="red",
        )
    )

    # Plot Obstacles
    for obstacle in state.obstacles:
        ax[0].add_patch(
            Rectangle(
                (obstacle[0] - state.obstacle_size, obstacle[1] - state.obstacle_size),
                state.obstacle_size * 2.0,
                state.obstacle_size * 2.0,
                color="black",
            )
        )

    # Plot Gems
    for gem_id in state.gems:
        if gem_id in state.at:
            ax[0].add_patch(
                Rectangle(
                    (
                        state.at[gem_id][0] - state.gem_size,
                        state.at[gem_id][1] - state.gem_size,
                    ),
                    state.gem_size * 2.0,
                    state.gem_size * 2.0,
                    color=CMAP(NORM(2.5 + float(gem_id[3]))),
                )
            )

    # Plot Held Objects
    ax[1].imshow(
        hold_data,
        cmap=CMAP,
        norm=NORM,
        origin="lower",
        interpolation="none",
        extent=[0.5, 1.5, 0.5, 8.5],
    )
