# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from beanmachine.facebook.goal_inference.planner.planner import (
    get_execution_path,
    StateNode,
)


def test_get_execution_path(dkg_domain, dkg_state_one):

    first_node = StateNode(dkg_state_one, None, [""])
    second_state = dkg_domain.execute(dkg_state_one, "down")
    second_node = StateNode(second_state, first_node, ["down"])
    third_state = dkg_domain.execute(second_state, "pickup", "key1")
    third_node = StateNode(third_state, second_node, ["pickup", "key1"])
    assert get_execution_path(third_node)[1] == [["down"], ["pickup", "key1"]]
