# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import dataclasses

import pytest

from beanmachine.facebook.goal_inference.doors_keys_gems.dkg_domain import DKGDomain

from beanmachine.facebook.goal_inference.doors_keys_gems.parser import parse


@pytest.fixture
def state():
    dkg_state = parse(
        "beanmachine/facebook/goal_inference/doors_keys_gems/test_problems/problem-1.pddl"
    )
    return dkg_state


@pytest.fixture
def domain():
    dkg_domain = DKGDomain()
    return dkg_domain


def test_wall(state, domain):

    for i in range(state.width):
        for j in range(state.height):
            assert domain.predicates["wall"](state, i + 1, j + 1) == (
                (i + 1, j + 1) in state.walls
            )


def test_door(state, domain):
    for i in range(state.width):
        for j in range(state.height):
            assert domain.predicates["door"](state, i + 1, j + 1) == (
                (i + 1, j + 1) in state.doors
            )


def test_obstacle(state, domain):
    for i in range(state.width):
        for j in range(state.height):
            assert domain.predicates["obstacle"](state, i + 1, j + 1) == (
                (i + 1, j + 1) in state.doors or (i + 1, j + 1) in state.walls
            )


def test_has(state, domain):

    for name in state.items:
        assert (name in state.has) == domain.predicates["has"](state, name)

    ### Assert No directions are held

    for direction in state.directions:
        assert not domain.predicates["has"](state, direction)

    ### items on ground are not held

    for name in state.at:
        assert not domain.predicates["has"](state, name)


def test_at(state, domain):

    for name in state.at:
        x_init = state.at[name][0]
        y_init = state.at[name][1]
        for i in range(state.width):
            for j in range(state.height):
                if i == x_init and j == y_init:
                    assert domain.predicates["at"](state, name, i, j)
                else:
                    assert not domain.predicates["at"](state, name, i, j)


#### Test Execution


def test_action_up(state, domain):

    state_two = domain.actions["up"].execute(state)

    assert state_two == state

    state_three = dataclasses.replace(state, y=state.y - 1)
    state_four = domain.actions["up"].execute(state_three)

    assert state_four.x == state_three.x
    assert state_four.y == state_three.y + 1


def test_action_down(state, domain):

    state_two = domain.execute(state, "down")

    assert state_two.x == state.x
    assert state_two.y == state.y - 1

    state_three = domain.execute(state_two, "down")

    assert state_three.x == state_two.x
    assert state_three.y == state_two.y - 1

    state_four = domain.execute(state_three, "down")

    assert state_four == state_three


def test_action_left(state, domain):

    state_two = domain.execute(state, "left")

    assert state_two == state

    state.walls.remove((2, 3))
    state_three = dataclasses.replace(state, x=2)

    state_four = domain.execute(state_three, "left")

    assert state_four.x == state_three.x - 1
    assert state_four.y == state_three.y


def test_action_right(state, domain):

    state_two = domain.execute(state, "right")

    assert state_two == state

    state.walls.remove((2, 3))
    state_three = dataclasses.replace(state, x=1)
    state_four = domain.execute(state_three, "right")

    assert state_four.x == state_three.x + 1
    assert state_four.y == state_three.y


def test_action_pickup(state, domain):

    state_two = domain.execute(state, "pickup", "key2")
    assert state_two == state

    state_three = dataclasses.replace(state, y=2)
    state_four = domain.execute(state_three, "pickup", "key2")
    assert state_four == state_three

    state_five = domain.execute(state_four, "pickup", "key1")
    assert "key1" in state_five.has


def test_action_unlock(state, domain):

    assert (2, 1) in state.doors

    state_two = dataclasses.replace(state, x=1, y=1)

    state_three = domain.execute(state_two, "unlock", "key1", "up")
    assert state_three == state_two

    state_four = domain.execute(state_three, "unlock", "key1", "left")
    assert state_four == state_three

    state_five = domain.execute(state_four, "unlock", "key1", "right")
    assert state_five == state_four

    state_six = dataclasses.replace(state_five, has={"key1": None})

    state_seven = domain.execute(state_six, "unlock", "key1", "up")
    assert state_seven == state_six

    state_eight = domain.execute(state_seven, "unlock", "key1", "left")
    assert state_eight == state_seven

    state_nine = domain.execute(state_eight, "unlock", "key1", "right")
    assert (2, 1) not in state_nine.doors


def test_get_possible_actions(state, domain):
    poss_actions = domain.get_possible_actions(state)
    assert poss_actions[0] == ["down"]
    assert len(poss_actions) == 1

    state = domain.execute(state, "down")
    poss_actions = domain.get_possible_actions(state)
    assert poss_actions[0] == ["up"]
    assert poss_actions[1] == ["down"]
    assert poss_actions[2] == ["pickup", "key1"]
    assert len(poss_actions) == 3

    state = domain.execute(state, "pickup", "key1")
    state = domain.execute(state, "down")
    poss_actions = domain.get_possible_actions(state)
    assert poss_actions[0] == ["up"]
    assert poss_actions[1] == ["unlock", "key1", "right"]
    assert len(poss_actions) == 2

    state = domain.execute(state, "unlock", "key1", "right")
    poss_actions = domain.get_possible_actions(state)
    assert poss_actions[0] == ["right"]
    assert poss_actions[1] == ["up"]
    assert len(poss_actions) == 2


def test_evaluate_goal(state, domain):
    assert not domain.evaluate_goal(state)
    state_two = dataclasses.replace(state, has={"key1": state.keys["key1"]})
    assert not domain.evaluate_goal(state_two)
    state_three = dataclasses.replace(state, has={"gem2": None})
    assert not domain.evaluate_goal(state_three)
    state_four = dataclasses.replace(state, has={"gem1": state.gems["gem1"]})
    assert domain.evaluate_goal(state_four)


def test_is_action_possible(state, domain):
    assert not domain.is_action_possible(state, "pickup", "gem1")
    assert domain.is_action_possible(state, "down")
    state = domain.execute(state, "down")
    assert domain.is_action_possible(state, "pickup", "key1")
