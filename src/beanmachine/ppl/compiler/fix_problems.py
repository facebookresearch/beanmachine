# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, Set, Tuple

import beanmachine.ppl.compiler.profiler as prof
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder

from beanmachine.ppl.compiler.devectorizer_transformer import vectorized_graph_fixer
from beanmachine.ppl.compiler.error_report import ErrorReport
from beanmachine.ppl.compiler.fix_additions import addition_fixer, sum_fixer
from beanmachine.ppl.compiler.fix_arithmetic import (
    log1mexp_fixer,
    neg_neg_fixer,
    negative_real_multiplication_fixer,
)
from beanmachine.ppl.compiler.fix_beta_conjugate_prior import (
    beta_bernoulli_conjugate_fixer,
    beta_binomial_conjugate_fixer,
)
from beanmachine.ppl.compiler.fix_bool_arithmetic import bool_arithmetic_fixer
from beanmachine.ppl.compiler.fix_bool_comparisons import bool_comparison_fixer
from beanmachine.ppl.compiler.fix_logsumexp import logsumexp_fixer
from beanmachine.ppl.compiler.fix_matrix_scale import (
    nested_matrix_scale_fixer,
    trivial_matmul_fixer,
)
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
    ancestors_first_graph_fixer,
    conditional_graph_fixer,
    fixpoint_graph_fixer,
    GraphFixer,
    GraphFixerResult,
    node_fixer_first_match,
    NodeFixer,
    sequential_graph_fixer,
)
from beanmachine.ppl.compiler.fix_requirements import requirements_fixer
from beanmachine.ppl.compiler.fix_transpose import identity_transpose_fixer
from beanmachine.ppl.compiler.fix_unsupported import (
    bad_matmul_reporter,
    unsupported_node_fixer,
    unsupported_node_reporter,
    untypable_node_reporter,
)
from beanmachine.ppl.compiler.lattice_typer import LatticeTyper


default_skip_optimizations: Set[str] = {
    "beta_bernoulli_conjugate_fixer",
    "beta_binomial_conjugate_fixer",
    "normal_normal_conjugate_fixer",
}


def arithmetic_graph_fixer(skip: Set[str]) -> GraphFixer:
    typer = LatticeTyper()

    def _arithmetic_graph_fixer(bmg: BMGraphBuilder) -> GraphFixerResult:
        node_fixers = [
            addition_fixer(bmg, typer),
            bool_arithmetic_fixer(bmg, typer),
            bool_comparison_fixer(bmg, typer),
            log1mexp_fixer(bmg, typer),
            logsumexp_fixer(bmg),
            multiary_addition_fixer(bmg),
            multiary_multiplication_fixer(bmg),
            neg_neg_fixer(bmg),
            negative_real_multiplication_fixer(bmg, typer),
            nested_matrix_scale_fixer(bmg),
            sum_fixer(bmg, typer),
            trivial_matmul_fixer(bmg, typer),
            unsupported_node_fixer(bmg, typer),
            identity_transpose_fixer(bmg, typer),
        ]
        node_fixers = [nf for nf in node_fixers if nf.__name__ not in skip]
        node_fixer = node_fixer_first_match(node_fixers)
        arith = ancestors_first_graph_fixer(typer, node_fixer)
        return fixpoint_graph_fixer(arith)(bmg)

    return _arithmetic_graph_fixer


_conjugacy_fixer_factories: List[Callable[[BMGraphBuilder], NodeFixer]] = [
    beta_bernoulli_conjugate_fixer,
    beta_binomial_conjugate_fixer,
    normal_normal_conjugate_fixer,
]


def conjugacy_graph_fixer(skip: Set[str]) -> GraphFixer:
    def _conjugacy_graph_fixer(bmg: BMGraphBuilder) -> GraphFixerResult:
        node_fixers = [
            f(bmg) for f in _conjugacy_fixer_factories if f.__name__ not in skip
        ]
        node_fixer = node_fixer_first_match(node_fixers)
        # TODO: Make the typer optional
        return ancestors_first_graph_fixer(LatticeTyper(), node_fixer)(bmg)

    return _conjugacy_graph_fixer


def fix_problems(
    bmg: BMGraphBuilder, skip_optimizations: Set[str] = default_skip_optimizations
) -> Tuple[BMGraphBuilder, ErrorReport]:
    current = bmg
    current._begin(prof.fix_problems)

    all_fixers = sequential_graph_fixer(
        [
            vectorized_graph_fixer(),
            arithmetic_graph_fixer(skip_optimizations),
            unsupported_node_reporter(),
            bad_matmul_reporter(),
            untypable_node_reporter(),
            conjugacy_graph_fixer(skip_optimizations),
            requirements_fixer,
            observations_fixer,
            conditional_graph_fixer(
                condition=lambda gb: gb._fix_observe_true, fixer=observe_true_fixer
            ),
        ]
    )
    current, _, errors = all_fixers(current)
    current._finish(prof.fix_problems)
    return current, errors
