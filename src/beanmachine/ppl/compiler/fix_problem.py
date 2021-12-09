# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Optional

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.error_report import BMGError, ErrorReport
from beanmachine.ppl.compiler.typer_base import TyperBase


class ProblemFixerBase(ABC):
    _bmg: BMGraphBuilder
    _typer: TyperBase
    errors: ErrorReport

    def __init__(self, bmg: BMGraphBuilder, typer: TyperBase) -> None:
        self._bmg = bmg
        self._typer = typer
        self.errors = ErrorReport()

    @abstractmethod
    def _needs_fixing(self, n: bn.BMGNode) -> bool:
        pass

    @abstractmethod
    def _get_replacement(self, n: bn.BMGNode) -> Optional[bn.BMGNode]:
        pass

    def _get_error(self, n: bn.BMGNode, index: int) -> Optional[BMGError]:
        # n.inputs[i] needed fixing but was unfixable. If that needs to
        # produce an error, do by overriding this method
        return None

    def fix_problems(self) -> None:
        replacements = {}
        reported = set()
        nodes = self._bmg.all_ancestor_nodes()
        for node in nodes:
            node_was_updated = False
            for i in range(len(node.inputs)):
                c = node.inputs[i]
                # Have we already replaced this input with something?
                # If so, no need to compute the replacement again.
                if c in replacements:
                    if node.inputs[i] is not replacements[c]:
                        node.inputs[i] = replacements[c]
                        node_was_updated = True
                    continue
                # Does the input need fixing at all?
                if not self._needs_fixing(c):
                    continue
                # The input needs fixing. Get the replacement.
                replacement = self._get_replacement(c)
                if replacement is not None:
                    replacements[c] = replacement
                    if node.inputs[i] is not replacement:
                        node.inputs[i] = replacement
                        node_was_updated = True
                    continue
                # It needed fixing but we did not succeed. Have we already
                # reported this error?  If so, no need to compute the error.
                if c in reported:
                    continue
                # Mark the node as having been error-reported, and emit
                # an error into the error report.
                reported.add(c)
                error = self._get_error(node, i)
                if error is not None:
                    self.errors.add_error(error)
            if node_was_updated:
                self._typer.update_type(node)
