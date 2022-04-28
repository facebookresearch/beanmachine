# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, Set

import beanmachine.ppl.compiler.profiler as prof
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.error_report import ErrorReport
from beanmachine.ppl.compiler.fix_additions import addition_fixer
from beanmachine.ppl.compiler.fix_beta_conjugate_prior import (
    beta_bernoulli_conjugate_fixer,
    beta_binomial_conjugate_fixer,
)
from beanmachine.ppl.compiler.fix_bool_arithmetic import bool_arithmetic_fixer
from beanmachine.ppl.compiler.fix_bool_comparisons import bool_comparison_fixer
from beanmachine.ppl.compiler.fix_logsumexp import logsumexp_fixer
from beanmachine.ppl.compiler.fix_matrix_scale import matrix_scale_fixer
from beanmachine.ppl.compiler.fix_multiary_ops import (
    multiary_addition_fixer,
    multiary_multiplication_fixer,
)
from beanmachine.ppl.compiler.fix_normal_conjugate_prior import (
    normal_normal_conjugate_fixer,
)
from beanmachine.ppl.compiler.fix_observations import observations_fixer
from beanmachine.ppl.compiler.fix_observe_true import observe_true_fixer
from beanmachine.ppl.compiler.fix_problem import (
    GraphFixer,
    NodeFixer,
    ancestors_first_graph_fixer,
    conditional_graph_fixer,
    node_fixer_first_match,
    sequential_graph_fixer,
)
from beanmachine.ppl.compiler.fix_requirements import requirements_fixer
from beanmachine.ppl.compiler.fix_unsupported import (
    unsupported_node_fixer,
    unsupported_node_reporter,
)
from beanmachine.ppl.compiler.fix_vectorized_models import (
    vectorized_observation_fixer,
    vectorized_operator_fixer,
)
from beanmachine.ppl.compiler.lattice_typer import LatticeTyper


# TODO[Walid]: Investigate ways to generalize transformations such as MatrixScale to
# work with multiary multiplications.

default_skip_optimizations: Set[str] = {
    "beta_bernoulli_conjugate_fixer",
    "beta_binomial_conjugate_fixer",
    "normal_normal_conjugate_fixer",
}

_arithmetic_fixer_factories: List[
    Callable[[BMGraphBuilder, LatticeTyper], NodeFixer]
] = [
    addition_fixer,
    bool_arithmetic_fixer,
    bool_comparison_fixer,
    matrix_scale_fixer,
    multiary_addition_fixer,
    multiary_multiplication_fixer,
    # TODO: logsumexp_fixer needs to come after multiary_addition_fixer right now;
    # when we make the arithmetic_graph_fixer attain a fixpoint then we can do
    # these fixers in any order.
    logsumexp_fixer,
    unsupported_node_fixer,
]


def arithmetic_graph_fixer(skip: Set[str], bmg: BMGraphBuilder) -> GraphFixer:
    typer = LatticeTyper()
    vector_ops = vectorized_operator_fixer(bmg)
    vector_obs = vectorized_observation_fixer(bmg)
    node_fixers = [
        f(bmg, typer) for f in _arithmetic_fixer_factories if f.__name__ not in skip
    ]
    node_fixer = node_fixer_first_match(node_fixers)
    arith = ancestors_first_graph_fixer(bmg, typer, node_fixer)
    # TODO: this should be a fixpoint combinator, not a sequence combinator
    return sequential_graph_fixer([vector_ops, vector_obs, arith])


_conjugacy_fixer_factories: List[Callable[[BMGraphBuilder], NodeFixer]] = [
    beta_bernoulli_conjugate_fixer,
    beta_binomial_conjugate_fixer,
    normal_normal_conjugate_fixer,
]


def conjugacy_graph_fixer(skip: Set[str], bmg: BMGraphBuilder) -> GraphFixer:
    node_fixers = [f(bmg) for f in _conjugacy_fixer_factories if f.__name__ not in skip]
    node_fixer = node_fixer_first_match(node_fixers)
    # TODO: Make the typer optional
    return ancestors_first_graph_fixer(bmg, LatticeTyper(), node_fixer)


def fix_problems(
    bmg: BMGraphBuilder, skip_optimizations: Set[str] = default_skip_optimizations
) -> ErrorReport:
    bmg._begin(prof.fix_problems)

    all_fixers = sequential_graph_fixer(
        [
            arithmetic_graph_fixer(skip_optimizations, bmg),
            unsupported_node_reporter(bmg),
            conjugacy_graph_fixer(skip_optimizations, bmg),
            requirements_fixer(bmg),
            observations_fixer(bmg),
            conditional_graph_fixer(bmg._fix_observe_true, observe_true_fixer(bmg)),
        ]
    )
    _, errors = all_fixers()
    bmg._finish(prof.fix_problems)
    return errors
