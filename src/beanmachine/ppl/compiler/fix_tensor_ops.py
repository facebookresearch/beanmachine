# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Optional

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.fix_problem import ProblemFixerBase
from beanmachine.ppl.compiler.typer_base import TyperBase


class TensorOpsFixer(ProblemFixerBase):
    """This class looks for reachable nodes in a graph which cannot be transformed
    to BMG because they contain tensor operations which have no representation. We
    rewrite those operations in a semantically equivalent form which can be
    represented."""

    # TODO: So far all we fix here is that a logsumexp node whose input is a
    # tensor becomes a logsumexp node whose inputs are the elements of the tensor.
    # We can do a lot better than that. For example, suppose we have:
    #
    # x = tensor([norm(1), norm(2)]) * 2.0
    # y = x.logsumexp(dim=0)
    #
    # Right now we cannot fix that because the input to logsumexp is not a tensor
    # node; it's a multiplication node. But we could first transform the graph to:
    #
    # x = tensor([norm(1) * 2.0, norm(2) * 2.0])
    # y = x.logsumexp(dim=0)
    #
    # and now it is in a form that we can fix further.
    #
    # Making this sort of change will require us to iterate on these changes until
    # we reach a fixpoint.

    def __init__(self, bmg: BMGraphBuilder, typer: TyperBase) -> None:
        ProblemFixerBase.__init__(self, bmg, typer)

    def _needs_fixing(self, n: bn.BMGNode) -> bool:
        return (
            isinstance(n, bn.LogSumExpNode)
            and len(n.inputs) == 1
            and isinstance(n.inputs[0], bn.TensorNode)
        )

    def _get_replacement(self, n: bn.BMGNode) -> Optional[bn.BMGNode]:
        # Suppose we have a model with a query on some samples:
        #
        # @function def f():
        #   return tensor([norm(), beta(), ... ]).logsumexp(dim=0)
        #
        # The graph accumulator will create a TensorNode containing the
        # SampleNodes, and the LogSumExpNode's input will be the tensor.
        # But we do not have a tensor-built-from-parts node in BMG.
        #
        # What we need to do is construct a new LogSumExpNode whose
        # inputs are the samples; the TensorNode will become orphaned.
        #

        assert isinstance(n, bn.LogSumExpNode)
        assert len(n.inputs) == 1
        t = n.inputs[0]
        assert isinstance(t, bn.TensorNode)
        # If the tensor is a singleton then logsumexp is
        # an identity.
        assert len(t.inputs) >= 1
        if len(t.inputs) == 1:
            return t.inputs[0]
        # Otherwise, we just make a LogSumExp whose inputs are the
        # tensor node elements.
        elements = t.inputs.inputs
        assert isinstance(elements, list)
        return self._bmg.add_logsumexp(*elements)
