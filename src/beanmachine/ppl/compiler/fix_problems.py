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
from beanmachine.ppl.compiler.fix_logsumexp import LogSumExpFixer
from beanmachine.ppl.compiler.fix_matrix_scale import matrix_scale_fixer
from beanmachine.ppl.compiler.fix_multiary_ops import (
    MultiaryAdditionFixer,
    MultiaryMultiplicationFixer,
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


def arithmetic_graph_fixer(bmg: BMGraphBuilder, typer: LatticeTyper) -> GraphFixer:
    node_fixer = node_fixer_first_match(
        [
            addition_fixer(bmg, typer),
            bool_arithmetic_fixer(bmg, typer),
            bool_comparison_fixer(bmg, typer),
            matrix_scale_fixer(bmg, typer),
            unsupported_node_fixer(bmg, typer),
        ]
    )
    return ancestors_first_graph_fixer(bmg, typer, node_fixer)


_standard_fixer_types: List[Callable] = [
    VectorizedModelFixer,
    arithmetic_graph_fixer,
    UnsupportedNodeReporter,
    MultiaryAdditionFixer,
    LogSumExpFixer,
    MultiaryMultiplicationFixer,
    BetaBernoulliConjugateFixer,
    BetaBinomialConjugateFixer,
    NormalNormalConjugateFixer,
    RequirementsFixer,
    ObservationsFixer,
]

default_skip_optimizations: Set[str] = {
    "BetaBernoulliConjugateFixer",
    "BetaBinomialConjugateFixer",
    "NormalNormalConjugateFixer",
}


def fix_problems(
    bmg: BMGraphBuilder, skip_optimizations: Set[str] = default_skip_optimizations
) -> ErrorReport:
    bmg._begin(prof.fix_problems)

    typer = LatticeTyper()
    fixer_types: List[Callable] = []
    for fixer_type in _standard_fixer_types:
        if fixer_type.__name__ not in skip_optimizations:
            fixer_types.append(fixer_type)
    errors = ErrorReport()
    if bmg._fix_observe_true:
        # Note: must NOT be +=, which would mutate _standard_fixer_types.
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
