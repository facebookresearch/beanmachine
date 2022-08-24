# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional, Union

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.copy_and_replace import (
    Cloner,
    copy_and_replace,
    NodeTransformer,
    TransformAssessment,
)
from beanmachine.ppl.compiler.error_report import ErrorReport, UnsizableNode
from beanmachine.ppl.compiler.fix_problem import GraphFixerResult
from beanmachine.ppl.compiler.sizer import Sizer, Unsized


class CopyGraph(NodeTransformer):
    def __init__(self, cloner: Cloner, sizer: Sizer):
        self.sizer = sizer
        self.cloner = cloner

    def assess_node(
        self, node: bn.BMGNode, original: BMGraphBuilder
    ) -> TransformAssessment:
        report = ErrorReport()
        transform = True
        try:
            size = self.sizer[node]
            if size == Unsized:
                transform = False
        except RuntimeError:
            transform = False
        if not transform:
            report.add_error(
                UnsizableNode(
                    node,
                    [self.sizer[input] for input in node.inputs.inputs],
                    self.cloner.bmg_original.execution_context.node_locations(node),
                )
            )
        return TransformAssessment(transform, report)

    def transform_node(
        self, node: bn.BMGNode, new_inputs: List[bn.BMGNode]
    ) -> Optional[Union[bn.BMGNode, List[bn.BMGNode]]]:
        return self.cloner.clone(
            node, [self.cloner.copy_context[p] for p in node.inputs.inputs]
        )


def copy(bmg_old: BMGraphBuilder) -> GraphFixerResult:
    bmg, errors = copy_and_replace(bmg_old, lambda c, s: CopyGraph(c, s))
    return bmg, True, errors
