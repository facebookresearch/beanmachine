# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import tempfile

from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from beanmachine.facebook.goal_inference.continuous_gems.cgem_definitions import (
    CGemState,
)

from beanmachine.facebook.goal_inference.doors_keys_gems.dkg_definitions import DKGState
from IPython.display import display, Image
from matplotlib.animation import FuncAnimation


def get_gem_inference_plot(
    inference_data: List[Dict[str, int]], frame: int
) -> Dict[str, List[float]]:
    """Formats posterior P(g|obs) from inference for plotting
    Applicable for doors, keys, and gems  / continuous gems problems

    Arguments:
        inference_data: Computation of the posterior P(goal|observations) at each time step
        frame: Time step to plot

    Returns:
        poster: P(goal|observations) for each time step until the current frame

    """
    curr_data = inference_data[: frame + 1]
    posterior = {}
    for goal_id in inference_data[0]:
        posterior[goal_id[1]] = []

    for time_step in range(len(curr_data)):
        for goal_id in curr_data[time_step]:
            posterior[goal_id[1]].append(curr_data[time_step][goal_id])

    return posterior


def get_gem_figure(
    state: Union[DKGState, CGemState],
    inference_data: Optional[List[Dict[str, int]]] = None,
    is_doors_keys_gems: bool = False,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Returns a figure with the current formatting
    Applicable for doors, keys, and gems  / continuous gems problems

    Arguments:
        state: State to get figure for
        inference_data: Computation of the posterior P(goal|observations) at each time step
        is_doors_keys_gems: Whether the problem is a doors, keys, and gems problem.

    Returns:
        fig: The figure object with the correct size/subplot settings
        axes: A list of Matplotlib Axes objects for further modification
    """

    fig = plt.figure(figsize=(1, 1))

    if inference_data is None:
        fig = plt.figure(figsize=(10, 5))
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
        ax3 = None
    else:
        fig = plt.figure(figsize=(7, 7))
        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 1, 2)

    plt.sca(ax1)
    plt.gca().set_title("Environment")

    if is_doors_keys_gems:

        plt.gca().set_xticks(np.arange(1, state.width + 1, 1))
        plt.gca().set_yticks(np.arange(1, state.height + 1, 1))

        plt.gca().set_xticks(np.arange(0.5, state.width + 1, 1), minor=True)
        plt.gca().set_yticks(np.arange(0.5, state.height + 1, 1), minor=True)

        plt.gca().grid(which="minor", color="k", linestyle="-", linewidth=2)
        plt.gca().grid(which="major", visible=False)

    else:

        plt.gca().set_xlim(0.0, state.width)
        plt.gca().set_ylim(0.0, state.height)

    plt.sca(ax2)

    plt.gca().set_title("Holding")
    plt.gca().set_xticks([])
    plt.gca().set_yticks(np.arange(1, 9, 1))

    plt.gca().set_xticks(np.arange(0.5, 2.0, 1), minor=True)
    plt.gca().set_yticks(np.arange(0.5, 9.0, 1), minor=True)

    plt.gca().grid(which="minor", color="k", linestyle="-", linewidth=2)
    plt.gca().grid(which="major", visible=False)

    axes = [ax1, ax2]

    if inference_data is not None:
        plt.sca(ax3)
        plt.gca().set_title("Inference")
        plt.gca().set_xlabel("Time Step")
        plt.gca().set_ylabel("Probability")
        plt.gca().set_ylim(0.0, 1.0)
        plt.gca().set_xlim(0.0, len(inference_data))
        plt.gca().set_xticks(np.arange(0, len(inference_data), 5))
        plt.gca().set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
        axes.append(ax3)

    return fig, axes


def format_gem_data(data: np.ndarray) -> np.ndarray:
    """Adjusts data matrix so it is the right orientation for imshow
    Applicable for doors, keys, and gems  / continuous gems problems

    Arguments:
        data: Input data array

    Returns:
        data: Modified data array
    """
    data = np.rot90(data, 1, (0, 1))
    data = np.rot90(data, 2, (0, 1))
    data = np.flip(data, axis=1)
    data += 0.5
    return data


def create_gif(
    fig: plt.Figure,
    animation_function: Callable,
    states: Union[List[DKGState], List[CGemState]],
    file_path: Optional[str],
):
    """Creates a gif for a series of states
    Applicable for doors, keys, and gems  / continuous gems problems

    Arguments:

        fig: Base matplotlib figure
        animation_function: Defines update of each frame of the gif
        states: Series of states to plot
        file_path: Optional location to save gif

    """

    anim_created = FuncAnimation(
        fig, animation_function, frames=len(states), interval=200
    )
    if file_path is None:
        with tempfile.TemporaryDirectory() as tmpdirname:
            anim_created.save(tmpdirname + "/tmp.gif", fps=10)
            display(Image(tmpdirname + "/tmp.gif", format="png"))
    else:
        anim_created.save(file_path, fps=10)
        display(Image(filename=file_path, format="png"))
