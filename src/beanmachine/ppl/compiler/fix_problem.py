# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABC, abstractmethod
from typing import Optional

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.error_report import BMGError, ErrorReport


class ProblemFixerBase(ABC):
    _bmg: BMGraphBuilder
    errors: ErrorReport

    def __init__(self, bmg: BMGraphBuilder) -> None:
        self._bmg = bmg
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
            for i in range(len(node.inputs)):
                c = node.inputs[i]
                # Have we already replaced this input with something?
                # If so, no need to compute the replacement again.
                if c in replacements:
                    node.inputs[i] = replacements[c]
                    continue
                # Does the input need fixing at all?
                if not self._needs_fixing(c):
                    continue
                # The input needs fixing. Get the replacement.
                replacement = self._get_replacement(c)
                if replacement is not None:
                    node.inputs[i] = replacement
                    replacements[c] = replacement
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
