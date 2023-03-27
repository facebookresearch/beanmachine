# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from beanmachine.facebook.goal_inference.planner.bfs import BFSPlanner


def test_bfs_generate_plan_dkg_problem_1(dkg_domain, dkg_state_one):

    bfs_solver = BFSPlanner(dkg_domain)
    plan, solved = bfs_solver.generate_plan(dkg_state_one)
    assert solved
    assert plan.actions == [
        ["down"],
        ["pickup", "key1"],
        ["down"],
        ["unlock", "key1", "right"],
        ["right"],
        ["right"],
        ["up"],
        ["up"],
        ["pickup", "gem1"],
    ]


def test_bfs_generate_plan_dkg_problem_2(dkg_domain, dkg_state_two):

    bfs_solver = BFSPlanner(dkg_domain)
    plan, solved = bfs_solver.generate_plan(dkg_state_two)
    assert solved
    assert len(plan.actions) == 14
