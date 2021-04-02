# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABC, abstractmethod

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder


class ProblemFixerBase(ABC):
    _bmg: BMGraphBuilder

    def __init__(self, bmg: BMGraphBuilder) -> None:
        self._bmg = bmg

    @abstractmethod
    def _needs_fixing(self, n: bn.BMGNode) -> bool:
        pass

    @abstractmethod
    def _get_replacement(self, n: bn.BMGNode) -> bn.BMGNode:
        pass

    def fix_problems(self) -> None:
        replacements = {}
        nodes = self._bmg._traverse_from_roots()
        for node in nodes:
            for i in range(len(node.inputs)):
                c = node.inputs[i]
                if c in replacements:
                    node.inputs[i] = replacements[c]
                    continue
                if self._needs_fixing(c):
                    replacement = self._get_replacement(c)
                    node.inputs[i] = replacement
                    replacements[c] = replacement
