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
    BetaBernoulliConjugateFixer,
    BetaBinomialConjugateFixer,
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
    NormalNormalConjugateFixer,
)
from beanmachine.ppl.compiler.fix_observations import ObservationsFixer
from beanmachine.ppl.compiler.fix_observe_true import ObserveTrueFixer
from beanmachine.ppl.compiler.fix_problem import (
    GraphFixer,
    node_fixer_first_match,
    ancestors_first_graph_fixer,
    NodeFixer,
)
from beanmachine.ppl.compiler.fix_requirements import RequirementsFixer
from beanmachine.ppl.compiler.fix_unsupported import (
    unsupported_node_fixer,
    UnsupportedNodeReporter,
)
from beanmachine.ppl.compiler.fix_vectorized_models import VectorizedModelFixer
from beanmachine.ppl.compiler.lattice_typer import LatticeTyper


# TODO[Walid]: Investigate ways to generalize transformations such as MatrixScale to
# work with multiary multiplications.

default_skip_optimizations: Set[str] = {
    "BetaBernoulliConjugateFixer",
    "BetaBinomialConjugateFixer",
    "NormalNormalConjugateFixer",
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


def arithmetic_graph_fixer(skip: Set[str]) -> Callable:
    def graph_fixer(bmg: BMGraphBuilder, typer: LatticeTyper) -> GraphFixer:
        node_fixers = [
            f(bmg, typer) for f in _arithmetic_fixer_factories if f.__name__ not in skip
        ]
        node_fixer = node_fixer_first_match(node_fixers)
        return ancestors_first_graph_fixer(bmg, typer, node_fixer)

    return graph_fixer


def fix_problems(
    bmg: BMGraphBuilder, skip_optimizations: Set[str] = default_skip_optimizations
) -> ErrorReport:
    bmg._begin(prof.fix_problems)

    # Functions with signature either
    # (BMGraphBuilder, Typer) -> GraphFixer
    # (BMGraphBuilder, Typer) -> ProblemFixer  # This will be refactored away
    graph_fixer_factories: List[Callable] = [
        VectorizedModelFixer,
        arithmetic_graph_fixer(skip_optimizations),
        UnsupportedNodeReporter,
        BetaBernoulliConjugateFixer,
        BetaBinomialConjugateFixer,
        NormalNormalConjugateFixer,
        RequirementsFixer,
        ObservationsFixer,
    ]

    typer = LatticeTyper()
    fixer_types: List[Callable] = [
        ft for ft in graph_fixer_factories if ft.__name__ not in skip_optimizations
    ]
    errors = ErrorReport()
    if bmg._fix_observe_true:
        fixer_types = fixer_types + [ObserveTrueFixer]
    for fixer_type in fixer_types:
        bmg._begin(fixer_type.__name__)
        fixer = fixer_type(bmg, typer)
        if hasattr(fixer, "fix_problems"):
            fixer.fix_problems()
            errors = fixer.errors
        else:
            _, errors = fixer()
        bmg._finish(fixer_type.__name__)

        if errors.any():
            break
    bmg._finish(prof.fix_problems)
    return errors
