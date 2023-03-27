# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from beanmachine.facebook.goal_inference.continuous_gems.cgem_parse import parse


def test_cgem_parse_problem_one():
    problem_one = parse(
        "beanmachine/facebook/goal_inference/continuous_gems/test_problems/problem-1.json"
    )
    assert problem_one.goal == ("has", "gem1")
    assert problem_one.width == 5.0
    assert problem_one.height == 5.0
    assert problem_one.gem_size == 0.25
    assert list(problem_one.at.keys()) == ["gem1", "gem2"]
    assert problem_one.at["gem1"] == (4.0, 0.5)
    assert problem_one.at["gem2"] == (0.5, 4.0)
    assert (2.0, 2.0) in problem_one.obstacles
    assert (2.0, 2.5) in problem_one.obstacles
    assert (4.0, 4.0) in problem_one.obstacles
    assert (0.5, 4.0) not in problem_one.obstacles
    assert problem_one.obstacle_size == 0.25
    assert problem_one.agent_size == 0.25
    assert problem_one.x == 1.0
    assert problem_one.y == 1.0
    assert problem_one.angle == 0.0


def test_cgem_parse_problem_two():
    problem_one = parse(
        "beanmachine/facebook/goal_inference/continuous_gems/test_problems/problem-2.json"
    )
    assert problem_one.goal == ("has", "gem1")
    assert problem_one.width == 5.0
    assert problem_one.height == 5.0
    assert problem_one.gem_size == 0.25
    assert list(problem_one.at.keys()) == ["gem1", "gem2"]
    assert problem_one.at["gem1"] == (0.5, 4.0)
    assert problem_one.at["gem2"] == (4.0, 4.0)
    assert (1.5, 0.5) in problem_one.obstacles
    assert (4.5, 0.5) in problem_one.obstacles
    assert (4.0, 4.0) not in problem_one.obstacles
    assert problem_one.obstacle_size == 0.5
    assert problem_one.agent_size == 0.25
    assert problem_one.x == 1.5
    assert problem_one.y == 1.5
    assert problem_one.angle == 0.0


def test_cgem_parse_problem_three():
    problem_one = parse(
        "beanmachine/facebook/goal_inference/continuous_gems/test_problems/problem-3.json"
    )
    assert problem_one.goal == ("has", "gem1")
    assert problem_one.width == 5.0
    assert problem_one.height == 5.0
    assert problem_one.gem_size == 0.25
    assert list(problem_one.at.keys()) == ["gem1", "gem2"]
    assert problem_one.at["gem1"] == (0.5, 4.5)
    assert problem_one.at["gem2"] == (4.5, 0.5)
    assert (0.25, 1.5) in problem_one.obstacles
    assert (2.0, 1.5) in problem_one.obstacles
    assert (0.321, 0.87) not in problem_one.obstacles
    assert problem_one.obstacle_size == 0.25
    assert problem_one.agent_size == 0.25
    assert problem_one.x == 2.5
    assert problem_one.y == 2.5
    assert problem_one.angle == 0.0
