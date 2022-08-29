# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing
from enum import Enum
from typing import Callable, Dict, List

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.broadcaster import broadcast_fnc
from beanmachine.ppl.compiler.copy_and_replace import (
    Cloner,
    copy_and_replace,
    NodeTransformer,
    TransformAssessment,
)
from beanmachine.ppl.compiler.error_report import ErrorReport, UnsizableNode
from beanmachine.ppl.compiler.fix_problem import (
    GraphFixer,
    GraphFixerResult,
    sequential_graph_fixer,
)
from beanmachine.ppl.compiler.sizer import is_scalar, Size, Sizer, Unsized
from beanmachine.ppl.compiler.tensorizer_transformer import Tensorizer

# elements in this list operate over tensors (all parameters are tensors) but do not necessarily produce tensors
_unary_tensor_ops = [
    bn.LogSumExpVectorNode,
    bn.MatrixExpNode,
    bn.MatrixSumNode,
    bn.TransposeNode,
    bn.ToPositiveRealMatrixNode,
    bn.ToRealMatrixNode,
    bn.CholeskyNode,
]

_binary_tensor_ops = [bn.ElementwiseMultiplyNode, bn.MatrixAddNode]

_tensor_constants = [
    bn.ConstantPositiveRealMatrixNode,
    bn.ConstantRealMatrixNode,
    bn.ConstantTensorNode,
    bn.UntypedConstantNode,
]

_tensor_valued_distributions = [bn.CategoricalNode, bn.DirichletNode]

_indexable_node_types = [
    bn.ColumnIndexNode,
    bn.ConstantTensorNode,
    bn.ElementwiseMultiplyNode,
    bn.IndexNode,
    bn.MatrixAddNode,
    bn.MatrixExpNode,
    bn.MatrixScaleNode,
    bn.MatrixMultiplicationNode,
    bn.SampleNode,
    bn.TensorNode,
    bn.ToMatrixNode,
    bn.UntypedConstantNode,
]


class ElementType(Enum):
    TENSOR = 1
    SCALAR = 2
    ANY = 3


class DevectorizeTransformation(Enum):
    YES = 1
    YES_WITH_MERGE = 2
    NO = 3


def _size_is_devectorizable(s: Size) -> bool:
    # TODO: support arbitrary devectorizing
    is_vector_or_matrix = not is_scalar(s) and len(s) <= 2
    return s != Unsized and is_vector_or_matrix


class CopyContext:
    def __init__(self):
        self.devectorized_nodes: Dict[bn.BMGNode, List[bn.BMGNode]] = {}
        self.clones: Dict[bn.BMGNode, bn.BMGNode] = {}


def _parameter_to_type_mm(node: bn.MatrixMultiplicationNode, index: int) -> ElementType:
    assert index == 0 or index == 1
    return ElementType.TENSOR


def _parameter_to_type_single_index(index: int) -> ElementType:
    if index == 0:
        return ElementType.TENSOR
    else:
        return ElementType.SCALAR


def _parameter_to_type_multi_index(node: bn.IndexNode, index: int) -> ElementType:
    if index == 0:
        return ElementType.TENSOR
    if len(node.inputs.inputs) > 2:
        return ElementType.TENSOR
    else:
        return ElementType.SCALAR


def _parameter_to_type_sample(node: bn.SampleNode, i: int) -> ElementType:
    if _tensor_valued_distributions.__contains__(type(node.operand)):
        return ElementType.TENSOR
    else:
        return ElementType.SCALAR


def _parameter_to_type_query(sizer: Sizer, node: bn.Query, index: int) -> ElementType:
    assert index == 0
    original_size = sizer[node]
    if original_size == Unsized or not is_scalar(original_size):
        return ElementType.TENSOR
    else:
        return ElementType.SCALAR


def _parameter_to_type_obs(node: bn.Observation, index: int) -> ElementType:
    # TODO: what is the expectation for Observations?
    # from the dirichlet tests it appears they can be tensors
    # for everything else, it looks like they must be scalars.
    # until I find out more, I'm implementing the solution that
    # enables all existing tests to pass
    sample = node.inputs.inputs[0]
    dist = sample.inputs.inputs[0]
    if _tensor_valued_distributions.__contains__(type(dist)):
        return ElementType.TENSOR
    else:
        return ElementType.SCALAR


def _parameter_to_type_matrix_scale(
    node: bn.MatrixScaleNode, index: int
) -> ElementType:
    if index == 0:
        return ElementType.SCALAR
    if index == 1:
        return ElementType.TENSOR
    else:
        raise ValueError(
            f"MatrixScale only has 2 inputs but index of {index} was provided"
        )


def _parameter_to_type_log_sum_exp(node: bn.LogSumExpNode, index: int) -> ElementType:
    return ElementType.SCALAR


def _parameter_to_type_torch_log_sum_exp(
    node: bn.LogSumExpTorchNode, index: int
) -> ElementType:
    assert index <= 2
    if index == 0:
        return ElementType.TENSOR
    else:
        return ElementType.SCALAR


class Devectorizer(NodeTransformer):
    def __init__(self, cloner: Cloner, sizer: Sizer):
        self.copy_context = CopyContext()
        self.cloner = cloner
        self.sizer = sizer
        self._parameter_to_type = {
            bn.MatrixMultiplicationNode: _parameter_to_type_mm,
            bn.ColumnIndexNode: lambda n, i: _parameter_to_type_single_index(i),
            bn.VectorIndexNode: lambda n, i: _parameter_to_type_single_index(i),
            bn.SampleNode: _parameter_to_type_sample,
            bn.LogSumExpNode: _parameter_to_type_log_sum_exp,
            bn.LogSumExpTorchNode: _parameter_to_type_torch_log_sum_exp,
            bn.Query: lambda n, i: _parameter_to_type_query(self.sizer, n, i),
            bn.Observation: _parameter_to_type_obs,
            bn.MatrixScaleNode: _parameter_to_type_matrix_scale,
            bn.IndexNode: _parameter_to_type_multi_index,
            bn.SwitchNode: self._parameter_to_type_switch,
        }

    def _parameter_to_type_switch(self, node: bn.SwitchNode, index: int) -> ElementType:
        if index == 0 or index % 2 == 1:
            return ElementType.SCALAR
        else:
            operand_of_concern = node.inputs.inputs[index]
            size = self.sizer[operand_of_concern]
            if size == Unsized:
                raise ValueError("every node should have been sized")
            if is_scalar(size):
                return ElementType.SCALAR
            else:
                return ElementType.TENSOR

    def __requires_element_type_at(self, node: bn.BMGNode, index: int) -> ElementType:
        node_type = type(node)
        if _tensor_valued_distributions.__contains__(node_type):
            return ElementType.TENSOR
        if _binary_tensor_ops.__contains__(node_type):
            return ElementType.TENSOR
        if self._parameter_to_type.__contains__(node_type):
            return self._parameter_to_type[node_type](node, index)
        if _unary_tensor_ops.__contains__(node_type):
            assert index == 0
            return ElementType.TENSOR
        else:
            return ElementType.SCALAR

    def __devectorize_transformation_type(
        self, node: bn.BMGNode
    ) -> DevectorizeTransformation:
        size = self.sizer[node]
        is_eligible_for_devectorize = _size_is_devectorizable(size) and not isinstance(
            node, bn.Query
        )
        if is_eligible_for_devectorize:
            # Determine if it needs to be split because the parent was split but could not be merged.
            # this is the case for example with a tensor version of normal. Suppose we draw a sample and that sample is the
            # operand to a matrix multiply. The sample doesn't need to be split because it doesn't consume tensors...it needs
            # to be split because its parent is no longer a tensor and cannot be a tensor
            def operand_is_no_longer_tensor(n: bn.BMGNode) -> bool:
                return self.copy_context.devectorized_nodes.__contains__(n) and not (
                    self.copy_context.clones.__contains__(n)
                )

            has_upstream_scatter_requirement = any(
                operand_is_no_longer_tensor(o) for o in node.inputs.inputs
            )

            # almost all distributions are scalar valued and cannot be merged
            if isinstance(node, bn.DistributionNode):
                if not _tensor_valued_distributions.__contains__(type(node)):
                    return DevectorizeTransformation.YES

            # it's possible that we need to split X because an operand has become an unmergable tensor
            # however, what if X is a tensor operand for a downstream tensor consumer? Then we need both
            has_merge_requirement = False
            has_downstream_scatter_requirement = False
            for consumer in node.outputs.items:
                index_of_me = next(
                    i
                    for i, producer in enumerate(consumer.inputs.inputs)
                    if producer == node
                )
                required_type = self.__requires_element_type_at(consumer, index_of_me)
                if required_type == ElementType.TENSOR:
                    has_merge_requirement = True
                elif required_type == ElementType.SCALAR:
                    has_downstream_scatter_requirement = True

            # it's possible that both tensor and scatter versions of the operands exist,
            # but we can't use it because a tensorized version of the op this node represents is unsupported
            node_does_not_support_tensors = all(
                self.__requires_element_type_at(node, i) == ElementType.SCALAR
                for i, _ in enumerate(node.inputs.inputs)
            ) and not _tensor_constants.__contains__(type(node))
            needs_devectorize = (
                has_upstream_scatter_requirement
                or node_does_not_support_tensors
                or has_downstream_scatter_requirement
            )
            if needs_devectorize and has_merge_requirement:
                return DevectorizeTransformation.YES_WITH_MERGE
            if needs_devectorize:
                return DevectorizeTransformation.YES
            if has_merge_requirement:
                return DevectorizeTransformation.NO

        return DevectorizeTransformation.NO

    def __get_clone_parents(
        self, node: bn.BMGNode
    ) -> List[typing.Union[bn.BMGNode, List[bn.BMGNode]]]:
        parents = []
        for j, p in enumerate(node.inputs.inputs):
            sz = self.sizer[p]
            if sz == Unsized:
                parent_was_tensor = False
            else:
                parent_was_tensor = not is_scalar(sz)

            required_element_type = self.__requires_element_type_at(node, j)
            needs_clone = required_element_type != ElementType.SCALAR or (
                not parent_was_tensor and required_element_type == ElementType.SCALAR
            )
            needs_devectorized = (
                required_element_type == ElementType.SCALAR and parent_was_tensor
            )
            if needs_clone:
                if self.copy_context.clones.__contains__(p):
                    parents.append(self.copy_context.clones[p])
                else:
                    raise ValueError("encountered a value not in the clone context")
            elif needs_devectorized:
                if self.copy_context.devectorized_nodes.__contains__(p):
                    parents.append(self.copy_context.devectorized_nodes[p])
                else:
                    raise ValueError("a vectorized parent was not found")
            else:
                raise NotImplementedError("This should be unreachable")

        return parents

    def __get_clone_parents_flat(self, node: bn.BMGNode) -> List[bn.BMGNode]:
        parents = []
        for p in node.inputs.inputs:
            if self.copy_context.clones.__contains__(p):
                parent = self.copy_context.clones[p]
                parents.append(parent)
            else:
                raise ValueError("a unit parent was not found")
        return parents

    def __flatten_parents(
        self, nd: bn.BMGNode, parents: List, creator: Callable
    ) -> List[bn.BMGNode]:
        return self.__flatten_parents_with_index(nd, parents, lambda i, s: creator(*s))

    def __flatten_parents_with_index(
        self, node: bn.BMGNode, parents: List, creator: Callable
    ) -> List[bn.BMGNode]:
        size = self.sizer[node]
        item_count = 1
        for i in range(0, len(size)):
            item_count *= size[i]
        elements: List[bn.BMGNode] = []
        broadcast: Dict[int, Callable] = {}
        for i, parent in enumerate(parents):
            if isinstance(parent, List):
                input_size = self.sizer[node.inputs.inputs[i]]
                broadbast_fnc_maybe = broadcast_fnc(input_size, size)
                if isinstance(broadbast_fnc_maybe, Callable):
                    broadcast[i] = broadbast_fnc_maybe
                else:
                    raise ValueError(
                        f"The size {input_size} cannot be broadcast to {size}"
                    )

        for i in range(0, item_count):
            reduced_parents = []
            for k, parent in enumerate(parents):
                if isinstance(parent, List):
                    new_index = broadcast[k](i)
                    reduced_parents.append(parent[new_index])
                else:
                    reduced_parents.append(parent)
            new_node = creator(i, reduced_parents)
            elements.append(new_node)
        return elements

    def _clone(self, node: bn.BMGNode) -> bn.BMGNode:
        n = self.cloner.clone(node, self.__get_clone_parents_flat(node))
        self.copy_context.clones[node] = n
        return n

    def __split(self, node: bn.BMGNode) -> List[bn.BMGNode]:
        size = self.sizer[node]
        dim = len(size)
        index_list = []
        # This code is a little confusing because BMG uses column-major matrices
        # and torch uses row-major tensors.  The Sizer always gives the size
        # that a graph node would be in *torch*, so if we have a Size([2, 3])
        # matrix node, that has two rows and three columns in torch, and would
        # be indexed first by row and then by column. But in BMG, that would
        # be two columns, three rows, and indexed by column first, then row.
        #
        # The practical upshot is: if we have, say, Size([3]) OR Size([1, 3])
        # then either way, we will have a one-column, three row BMG node, and
        # therefore we only need a single level of indexing.
        n = self._clone(node)
        if dim == 0:
            # If we have just a single value then there's no indexing required.
            index_list.append(n)
        elif dim == 1:
            for i in range(0, size[0]):
                ci = self.cloner.bmg.add_constant(i)
                ni = self.cloner.bmg.add_index(n, ci)
                index_list.append(ni)
        elif size[0] == 1:
            assert dim == 2
            for i in range(0, size[1]):
                ci = self.cloner.bmg.add_constant(i)
                ni = self.cloner.bmg.add_index(n, ci)
                index_list.append(ni)
        else:
            # We need two levels of indexing.
            assert dim == 2
            for i in range(0, size[0]):
                ci = self.cloner.bmg.add_constant(i)
                ni = self.cloner.bmg.add_index(n, ci)
                for j in range(0, size[1]):
                    cj = self.cloner.bmg.add_constant(j)
                    nij = self.cloner.bmg.add_index(ni, cj)
                    index_list.append(nij)
        return index_list

    def __scatter(self, node: bn.BMGNode) -> List[bn.BMGNode]:
        parents = self.__get_clone_parents(node)
        if isinstance(node, bn.SampleNode):
            new_nodes = self.__flatten_parents(
                node, parents, self.cloner.bmg.add_sample
            )
            return new_nodes
        if isinstance(node, bn.OperatorNode) or isinstance(node, bn.DistributionNode):
            return self.__flatten_parents(
                node, parents, self.cloner.node_factories[type(node)]
            )
        if isinstance(node, bn.Observation):
            dim = len(node.value.size())
            values = []
            if dim == 0:
                values.append(node.value.item())
            elif dim == 1:
                for i in range(0, node.value.size()[0]):
                    values.append(node.value[i])
            else:
                assert dim == 2
                for i in range(0, node.value.size()[0]):
                    for j in range(0, node.value.size()[1]):
                        values.append(node.value[i][j])

            return self.__flatten_parents_with_index(
                node,
                parents,
                lambda i, s: self.__add_observation(s, i, values),
            )
        else:
            raise NotImplementedError()

    def __add_observation(
        self, inputs: List[bn.BMGNode], i: int, value: List
    ) -> bn.Observation:
        assert len(inputs) == 1
        sample = inputs[0]
        if isinstance(sample, bn.SampleNode):
            return self.cloner.bmg.add_observation(sample, value[i])
        else:
            raise ValueError("expected a sample as a parent to an observation")

    def _devectorize(self, node: bn.BMGNode) -> List[bn.BMGNode]:
        # there are two ways to devectorize a node: (1) we can scatter it or (2) we can split it (clone and index)
        is_sample_of_scalar_dist = isinstance(
            node, bn.SampleNode
        ) and not _tensor_valued_distributions.__contains__(node.operand)
        not_indexable = not _indexable_node_types.__contains__(type(node))
        if not_indexable or is_sample_of_scalar_dist:
            return self.__scatter(node)
        else:
            return self.__split(node)

    def assess_node(
        self, node: bn.BMGNode, original: BMGraphBuilder
    ) -> TransformAssessment:
        if self.sizer[node] == Unsized:
            report = ErrorReport()
            report.add_error(
                UnsizableNode(
                    node,
                    [self.sizer[p] for p in node.inputs.inputs],
                    original.execution_context.node_locations(node),
                )
            )
            return TransformAssessment(False, report)
        return TransformAssessment(True, ErrorReport())

    # a node is either replaced 1-1, 1-many, or deleted
    def transform_node(
        self, node: bn.BMGNode, new_inputs: List[bn.BMGNode]
    ) -> typing.Optional[typing.Union[bn.BMGNode, List[bn.BMGNode]]]:
        transform_type = self.__devectorize_transformation_type(node)
        if transform_type == DevectorizeTransformation.YES:
            image = self._devectorize(node)
            self.copy_context.devectorized_nodes[node] = image
        elif transform_type == DevectorizeTransformation.NO:
            image = self._clone(node)
            self.copy_context.clones[node] = image
        elif transform_type == DevectorizeTransformation.YES_WITH_MERGE:
            image = self._devectorize(node)
            self.copy_context.devectorized_nodes[node] = image
            if not self.copy_context.clones.__contains__(node):
                tensor = self.cloner.bmg.add_tensor(self.sizer[node], *image)
                self.copy_context.clones[node] = tensor
            image = self.copy_context.clones[node]
        else:
            raise NotImplementedError(
                "a new type of transformation type was introduced but never implemented"
            )
        return image


def vectorized_graph_fixer() -> GraphFixer:
    def _tensorize(bmg_old: BMGraphBuilder) -> GraphFixerResult:
        bmg, errors = copy_and_replace(bmg_old, lambda c, s: Tensorizer(c, s))
        return bmg, True, errors

    def _detensorize(bmg_old: BMGraphBuilder) -> GraphFixerResult:
        bmg, errors = copy_and_replace(bmg_old, lambda c, s: Devectorizer(c, s))
        return bmg, True, errors

    return sequential_graph_fixer([_tensorize, _detensorize])
