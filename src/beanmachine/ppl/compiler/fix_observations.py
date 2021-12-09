# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_types import (
    Boolean,
    Natural,
    PositiveReal,
    Probability,
    Real,
    Untypable,
    supremum,
    type_of_value,
)
from beanmachine.ppl.compiler.error_report import ErrorReport, ImpossibleObservation
from beanmachine.ppl.compiler.lattice_typer import LatticeTyper


class ObservationsFixer:
    """This class takes a Bean Machine Graph builder and attempts to
    fix violations of BMG type system requirements in observation nodes.
    It also finds observations that are impossible -- say, an observation
    that a Boolean node is -3.14 -- and reports them as errors."""

    errors: ErrorReport
    bmg: BMGraphBuilder
    _typer: LatticeTyper

    def __init__(self, bmg: BMGraphBuilder, typer: LatticeTyper) -> None:
        self.errors = ErrorReport()
        self.bmg = bmg
        self._typer = typer

    def fix_problems(self) -> None:
        for o in self.bmg.all_observations():
            v = o.value
            value_type = type_of_value(v)
            assert value_type != Untypable
            sample_type = self._typer[o.observed]
            assert sample_type != Untypable
            if supremum(value_type, sample_type) != sample_type:
                self.errors.add_error(ImpossibleObservation(o, sample_type))
            elif sample_type == Boolean:
                o.value = bool(v)
            elif sample_type == Natural:
                o.value = int(v)
            elif sample_type in {Probability, PositiveReal, Real}:
                o.value = float(v)
            else:
                # TODO: How should we deal with observations of
                # TODO: matrix-valued samples?
                pass
            # TODO: Handle the case where there are two inconsistent
            # TODO: observations of the same sample
