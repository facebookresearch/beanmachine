# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_types import (
    Boolean,
    Natural,
    NegativeReal,
    PositiveReal,
    Probability,
    Real,
    supremum,
    type_of_value,
    Untypable,
)
from beanmachine.ppl.compiler.error_report import ErrorReport, ImpossibleObservation
from beanmachine.ppl.compiler.fix_problem import GraphFixer
from beanmachine.ppl.compiler.lattice_typer import LatticeTyper


def observations_fixer(bmg: BMGraphBuilder) -> GraphFixer:
    """This fixer attempts to fix violations of BMG type system requirements
    in observation nodes.
    It also finds observations that are impossible -- say, an observation
    that a Boolean node is -3.14 -- and reports them as errors."""

    def fixer():
        typer = LatticeTyper()
        errors = ErrorReport()
        made_progress = False
        for o in bmg.all_observations():
            v = o.value
            value_type = type_of_value(v)
            assert value_type != Untypable
            sample_type = typer[o.observed]
            assert sample_type != Untypable
            if supremum(value_type, sample_type) != sample_type:
                errors.add_error(ImpossibleObservation(o, sample_type))
            elif sample_type == Boolean and not isinstance(v, bool):
                o.value = bool(v)
                made_progress = True
            elif sample_type == Natural and not isinstance(v, int):
                o.value = int(v)
                made_progress = True
            elif sample_type in {
                Probability,
                PositiveReal,
                NegativeReal,
                Real,
            } and not isinstance(v, float):
                o.value = float(v)
                made_progress = True
            else:
                # TODO: How should we deal with observations of
                # TODO: matrix-valued samples?
                pass
            # TODO: Handle the case where there are two inconsistent
            # TODO: observations of the same sample
        return made_progress, errors

    return fixer
