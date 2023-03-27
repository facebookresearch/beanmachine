# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from beanmachine.facebook.goal_inference.doors_keys_gems.parser import parse


def test_state_load_problem_1():
    state = parse(
        "beanmachine/facebook/goal_inference/doors_keys_gems/test_problems/problem-1.pddl"
    )
    ### walls
    assert len(state.has) == 0
    assert (2, 3) in state.walls
    assert (2, 2) in state.walls
    ### doors
    assert (2, 1) in state.doors
    ### items
    assert "key1" in state.at and state.at[("key1")] == (1, 2)
    assert "gem1" in state.at and state.at[("gem1")] == (3, 3)

    ### boundaries
    assert state.width == 3
    assert state.height == 3
    ### agent
    assert state.x == 1
    assert state.y == 3

    ### xdiff / ydiff
    assert state.xdiff["right"] == 1
    assert state.xdiff["left"] == -1
    assert state.xdiff["up"] == 0
    assert state.xdiff["down"] == 0
    assert state.ydiff["right"] == 0
    assert state.ydiff["left"] == 0
    assert state.ydiff["up"] == 1
    assert state.ydiff["down"] == -1

    assert state.goal == ("has", "gem1")


def test_state_load_problem_2():
    state = parse(
        "beanmachine/facebook/goal_inference/doors_keys_gems/test_problems/problem-2.pddl"
    )

    ### walls
    assert len(state.has) == 0
    assert (2, 2) in state.walls
    assert (2, 3) in state.walls
    assert (2, 4) in state.walls
    assert (4, 2) in state.walls
    assert (4, 3) in state.walls
    assert (4, 4) in state.walls
    ### doors
    assert (4, 1) in state.doors
    assert (4, 5) in state.doors

    ### items
    assert "key1" in state.at and state.at[("key1")] == (3, 3)
    assert "gem1" in state.at and state.at[("gem1")] == (5, 2)
    assert "gem2" in state.at and state.at[("gem2")] == (5, 4)

    ### boundaries
    assert state.width == 5
    assert state.height == 5
    ### agent
    assert state.x == 1
    assert state.y == 3

    ### xdiff / ydiff
    assert state.xdiff["right"] == 1
    assert state.xdiff["left"] == -1
    assert state.xdiff["up"] == 0
    assert state.xdiff["down"] == 0
    assert state.ydiff["right"] == 0
    assert state.ydiff["left"] == 0
    assert state.ydiff["up"] == 1
    assert state.ydiff["down"] == -1

    assert state.goal == ("has", "gem1")


def test_state_load_problem_3():
    state = parse(
        "beanmachine/facebook/goal_inference/doors_keys_gems/test_problems/problem-3.pddl"
    )

    ### walls
    assert len(state.has) == 0
    assert (1, 2) in state.walls
    assert (2, 2) in state.walls
    assert (3, 2) in state.walls
    assert (5, 1) in state.walls
    assert (5, 2) in state.walls
    assert (5, 3) in state.walls

    assert (5, 4) in state.walls
    assert (7, 2) in state.walls
    assert (7, 3) in state.walls
    assert (7, 4) in state.walls
    assert (7, 5) in state.walls
    assert (7, 6) in state.walls
    assert (2, 4) in state.walls
    assert (2, 5) in state.walls
    assert (2, 6) in state.walls
    assert (3, 6) in state.walls
    assert (4, 6) in state.walls
    assert (5, 6) in state.walls
    assert (6, 6) in state.walls
    assert (7, 6) in state.walls
    assert (4, 8) in state.walls
    assert (6, 7) in state.walls
    ### doors
    assert (4, 7) in state.doors
    assert (5, 5) in state.doors
    assert (8, 6) in state.doors

    ### items
    assert "key1" in state.at and state.at[("key1")] == (1, 7)
    assert "key2" in state.at and state.at[("key2")] == (7, 7)
    assert "gem1" in state.at and state.at[("gem1")] == (1, 8)
    assert "gem2" in state.at and state.at[("gem2")] == (8, 8)
    assert "gem3" in state.at and state.at[("gem3")] == (8, 1)

    ### boundaries
    assert state.width == 8
    assert state.height == 8
    ### agent
    assert state.x == 1
    assert state.y == 1

    ### xdiff / ydiff
    assert state.xdiff["right"] == 1
    assert state.xdiff["left"] == -1
    assert state.xdiff["up"] == 0
    assert state.xdiff["down"] == 0
    assert state.ydiff["right"] == 0
    assert state.ydiff["left"] == 0
    assert state.ydiff["up"] == 1
    assert state.ydiff["down"] == -1

    assert state.goal == ("has", "gem3")


def test_state_load_problem_4():
    state = parse(
        "beanmachine/facebook/goal_inference/doors_keys_gems/test_problems/problem-4.pddl"
    )

    ### walls
    assert len(state.has) == 0
    assert (4, 1) in state.walls
    assert (5, 1) in state.walls
    assert (6, 1) in state.walls
    assert (2, 2) in state.walls
    assert (3, 2) in state.walls
    assert (4, 2) in state.walls
    assert (6, 2) in state.walls
    assert (7, 2) in state.walls
    assert (8, 2) in state.walls
    assert (2, 3) in state.walls
    assert (8, 3) in state.walls
    assert (2, 4) in state.walls
    assert (4, 4) in state.walls
    assert (6, 4) in state.walls
    assert (8, 4) in state.walls
    assert (2, 5) in state.walls
    assert (4, 5) in state.walls
    assert (6, 5) in state.walls
    assert (8, 5) in state.walls
    assert (2, 6) in state.walls
    assert (4, 6) in state.walls
    assert (5, 6) in state.walls
    assert (6, 6) in state.walls
    assert (8, 6) in state.walls
    assert (2, 7) in state.walls
    assert (4, 7) in state.walls
    assert (6, 7) in state.walls
    assert (8, 7) in state.walls
    assert (2, 8) in state.walls
    assert (3, 8) in state.walls
    assert (4, 8) in state.walls
    assert (6, 8) in state.walls
    assert (8, 8) in state.walls
    ### doors
    assert (5, 4) in state.doors
    assert (7, 8) in state.doors

    ### items
    assert "key1" in state.at and state.at[("key1")] == (5, 7)
    assert "key2" in state.at and state.at[("key2")] == (5, 9)
    assert "gem1" in state.at and state.at[("gem1")] == (7, 1)
    assert "gem2" in state.at and state.at[("gem2")] == (5, 2)
    assert "gem3" in state.at and state.at[("gem3")] == (5, 5)

    ### boundaries
    assert state.width == 9
    assert state.height == 9
    ### agent
    assert state.x == 3
    assert state.y == 1

    ### xdiff / ydiff
    assert state.xdiff["right"] == 1
    assert state.xdiff["left"] == -1
    assert state.xdiff["up"] == 0
    assert state.xdiff["down"] == 0
    assert state.ydiff["right"] == 0
    assert state.ydiff["left"] == 0
    assert state.ydiff["up"] == 1
    assert state.ydiff["down"] == -1

    assert state.goal == ("has", "gem3")
