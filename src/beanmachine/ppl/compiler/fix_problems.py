# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Set, Type

import beanmachine.ppl.compiler.profiler as prof
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.error_report import ErrorReport
from beanmachine.ppl.compiler.fix_additions import AdditionFixer
from beanmachine.ppl.compiler.fix_beta_conjugate_prior import (
    BetaBernoulliConjugateFixer,
    BetaBinomialConjugateFixer,
)
from beanmachine.ppl.compiler.fix_bool_arithmetic import BoolArithmeticFixer
from beanmachine.ppl.compiler.fix_bool_comparisons import BoolComparisonFixer
from beanmachine.ppl.compiler.fix_logsumexp import LogSumExpFixer
from beanmachine.ppl.compiler.fix_matrix_scale import MatrixScaleFixer
from beanmachine.ppl.compiler.fix_multiary_ops import (
    MultiaryAdditionFixer,
    MultiaryMultiplicationFixer,
)
from beanmachine.ppl.compiler.fix_normal_conjugate_prior import (
    NormalNormalConjugateFixer,
)
from beanmachine.ppl.compiler.fix_observations import ObservationsFixer
from beanmachine.ppl.compiler.fix_observe_true import ObserveTrueFixer
from beanmachine.ppl.compiler.fix_requirements import RequirementsFixer
from beanmachine.ppl.compiler.fix_unsupported import UnsupportedNodeFixer
from beanmachine.ppl.compiler.fix_vectorized_models import VectorizedModelFixer
from beanmachine.ppl.compiler.lattice_typer import LatticeTyper


# Some notes on ordering:
#
# AdditionFixer needs to run before RequirementsFixer. Why?
# The requirement fixing pass runs from leaves to roots, inserting
# conversions as it goes. If we have add(1, negate(p)) then we need
# to turn that into complement(p), but if we process the add *after*
# we process the negate(p) then we will already have generated
# add(1, negate(to_real(p)).  Better to turn it into complement(p)
# and orphan the negate(p) early.
#
# TODO: Add other notes on ordering constraints here.

# TODO[Walid]: Investigate ways to generalize transformations such as MatrixScale to
# work with multiary multiplications.

_standard_fixer_types: List[Type] = [
    VectorizedModelFixer,
    BoolArithmeticFixer,
    AdditionFixer,
    BoolComparisonFixer,
    UnsupportedNodeFixer,
    MatrixScaleFixer,
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
    fixer_types: List[Type] = []
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
        fixer.fix_problems()
        bmg._finish(fixer_type.__name__)
        errors = fixer.errors
        if errors.any():
            break
    bmg._finish(prof.fix_problems)
    return errors
