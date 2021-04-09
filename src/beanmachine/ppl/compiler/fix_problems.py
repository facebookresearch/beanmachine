# Copyright (c) Facebook, Inc. and its affiliates.

from typing import List, Type

import beanmachine.ppl.compiler.profiler as prof
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.error_report import ErrorReport
from beanmachine.ppl.compiler.fix_additions import AdditionFixer
from beanmachine.ppl.compiler.fix_bool_arithmetic import BoolArithmeticFixer
from beanmachine.ppl.compiler.fix_bool_comparisons import BoolComparisonFixer
from beanmachine.ppl.compiler.fix_multiary_ops import MultiaryOperatorFixer
from beanmachine.ppl.compiler.fix_observations import ObservationsFixer
from beanmachine.ppl.compiler.fix_observe_true import ObserveTrueFixer
from beanmachine.ppl.compiler.fix_requirements import RequirementsFixer
from beanmachine.ppl.compiler.fix_tensor_ops import TensorOpsFixer
from beanmachine.ppl.compiler.fix_unsupported import UnsupportedNodeFixer


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

_standard_fixer_types: List[Type] = [
    TensorOpsFixer,
    BoolArithmeticFixer,
    AdditionFixer,
    BoolComparisonFixer,
    UnsupportedNodeFixer,
    MultiaryOperatorFixer,
    RequirementsFixer,
    ObservationsFixer,
]


def fix_problems(bmg: BMGraphBuilder, fix_observe_true: bool = False) -> ErrorReport:
    bmg._begin(prof.fix_problems)
    fixer_types: List[Type] = _standard_fixer_types
    errors = ErrorReport()
    if fix_observe_true:
        # Note: must NOT be +=, which would mutate _standard_fixer_types.
        fixer_types = fixer_types + [ObserveTrueFixer]
    for fixer_type in fixer_types:
        bmg._begin(fixer_type.__name__)
        fixer = fixer_type(bmg)
        fixer.fix_problems()
        bmg._finish(fixer_type.__name__)
        errors = fixer.errors
        if errors.any():
            break
    bmg._finish(prof.fix_problems)
    return errors
