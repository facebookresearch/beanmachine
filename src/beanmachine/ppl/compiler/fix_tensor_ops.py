# Copyright (c) Facebook, Inc. and its affiliates.

from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_nodes import BMGNode, LogSumExpNode, TensorNode


def _is_fixable_logsumexp(n: BMGNode) -> bool:
    return (
        isinstance(n, LogSumExpNode)
        and len(n.inputs) == 1
        and isinstance(n.inputs[0], TensorNode)
    )


class TensorOpsFixer:
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

    bmg: BMGraphBuilder

    def __init__(self, bmg: BMGraphBuilder) -> None:
        self.bmg = bmg

    def _fix_logsumexp(self, n: LogSumExpNode) -> BMGNode:
        assert len(n.inputs) == 1
        t = n.inputs[0]
        assert isinstance(t, TensorNode)
        # If the tensor is a singleton then logsumexp is
        # an identity.
        assert len(t.inputs) >= 1
        if len(t.inputs) == 1:
            return t.inputs[0]
        # Otherwise, we just make a LogSumExp whose inputs are the
        # tensor node elements.
        elements = t.inputs.inputs
        assert isinstance(elements, list)
        return self.bmg.add_logsumexp(*elements)

    def fix_tensor_ops(self) -> None:
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
        replacements = {}
        nodes = self.bmg._traverse_from_roots()
        for node in nodes:
            for i in range(len(node.inputs)):
                c = node.inputs[i]
                if c in replacements:
                    node.inputs[i] = replacements[c]
                    continue
                if _is_fixable_logsumexp(c):
                    assert isinstance(c, LogSumExpNode)
                    replacement = self._fix_logsumexp(c)
                    node.inputs[i] = replacement
                    replacements[c] = replacement
