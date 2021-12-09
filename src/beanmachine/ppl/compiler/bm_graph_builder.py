# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Callable, Dict, List, Optional

import beanmachine.ppl.compiler.bmg_nodes as bn
import beanmachine.ppl.compiler.bmg_types as bt
import beanmachine.ppl.compiler.profiler as prof
import numpy as np
import torch
import torch.distributions as dist
from beanmachine.ppl.compiler.bmg_nodes import BMGNode, ConstantNode
from beanmachine.ppl.compiler.hint import log1mexp, math_log1mexp
from beanmachine.ppl.utils.memoize import memoize


_standard_normal = dist.Normal(0.0, 1.0)


def phi(x: Any) -> Any:
    return _standard_normal.cdf(x)


supported_bool_types = {bool, np.bool_}
supported_float_types = {np.longdouble, np.float16, np.float32, np.float64, float}
supported_int_types = {np.int16, np.int32, np.int64, np.int8, np.longlong}
supported_int_types |= {np.uint16, np.uint32, np.uint64, np.uint8, np.ulonglong, int}


class BMGraphBuilder:

    # ####
    # #### State and initialization
    # ####

    # We keep a list of all the nodes in the graph and associate a unique
    # integer with each.

    # TODO: The original idea was to use these integers when generating code
    # that constructs the graph, or DOT files that display the graph.
    # However, the integer generated is ordered according to when the node
    # was created, which is not necessarily the order in which we would
    # like to enumerate them for code generation purposes.
    #
    # We have therefore changed the code generation process to do a deterministic
    # topological sort of the nodes, and then number them in topological sort
    # order when emitting code; that way the code is generated so that each node
    # is numbered in the order it appears in the code. This is more pleasant
    # to read and understand, but the real benefit is that it makes the test
    # cases more stable and easier to verify.
    #
    # We can replace this dictionary with an unordered set; consider doing so.

    _nodes: Dict[BMGNode, int]
    _node_counter: int

    # This allows us to turn on a special problem-fixing pass to help
    # work around problems under investigation.

    _fix_observe_true: bool = False

    _pd: Optional[prof.ProfilerData]

    def __init__(self) -> None:
        self._nodes = {}
        self._node_counter = 0
        self._pd = None

    def _begin(self, s: str) -> None:
        pd = self._pd
        if pd is not None:
            pd.begin(s)

    def _finish(self, s: str) -> None:
        pd = self._pd
        if pd is not None:
            pd.finish(s)

    # ####
    # #### Node creation and accumulation
    # ####

    # This code is called while the lifted program executes.
    #
    # The "add" methods unconditionally create a new graph node
    # and add to the builder *if it does not already exist*.
    # By memoizing almost all the "add" methods we ensure that
    # the graph is deduplicated automatically.
    #
    # The "handle" methods, in contrast, conditionally create new
    # graph nodes only when required because an operation on a
    # stochastic value must be accumulated into the graph.
    #
    # For example, if we have a call to handle_addition where all
    # operands are ordinary floats (or constant graph nodes)
    # then there is no need to add a new node to the graph. But
    # if we have an addition of 1.0 to a stochastic node -- perhaps
    # a sample node, or perhaps some other graph node that eventually
    # involves a sample node -- then we need to construct a new
    # addition node, which is then returned and becomes the value
    # manipulated by the executing lifted program.
    #
    # TODO: The code in the "handle" methods which folds operations
    # on constant nodes and regular values is a holdover from an
    # earlier prototyping stage in which all values were lifted to
    # graph nodes. These scenarios should now be impossible, and we
    # should take a work item to remove this now-unnecessary code.

    def add_node(self, node: BMGNode) -> BMGNode:
        # TODO: This should be private
        """This adds a node we've recently created to the node set;
        it maintains the invariant that all the input nodes are also added."""
        assert node not in self._nodes
        for i in node.inputs:
            assert i in self._nodes
        self._nodes[node] = self._node_counter
        self._node_counter += 1
        return node

    def remove_leaf(self, node: BMGNode) -> None:
        # TODO: This is only used to remove an observation; restrict it
        # accordingly. Particularly because this code is not correct with
        # respect to the memoizers. We could add a node, memoize it,
        # remove, it, add it again, and the memoizer would hand the node
        # back without adding it to the graph.

        """This removes a leaf node from the builder, and ensures that the
        output edges of all its input nodes are removed as well."""
        if not node.is_leaf:
            raise ValueError("remove_leaf requires a leaf node")
        if node not in self._nodes:
            raise ValueError("remove_leaf called with node from wrong builder")
        for i in node.inputs.inputs:
            i.outputs.remove_item(node)
        del self._nodes[node]

    def remove_node(self, node: BMGNode) -> None:
        # TODO: This is only used to remove observation sample node that
        # are factored into posterior computation; restrict it
        # accordingly. Particularly because this code is not correct with
        # respect to the memoizers. We could add a node, memoize it,
        # remove, it, add it again, and the memoizer would hand the node
        # back without adding it to the graph.

        """This removes a node from the builder, and ensures that the
        output edges of all its input nodes are removed as well."""
        if node not in self._nodes:
            raise ValueError("remove_node called with node from wrong builder")
        for i in node.inputs.inputs:
            i.outputs.remove_item(node)
        del self._nodes[node]

    # ####
    # #### Graph accumulation for constant values
    # ####

    # This code handles creating nodes for ordinary values such as
    # floating point values and tensors created during the execution
    # of the lifted program. We only create graph nodes for an ordinary
    # value when that value is somehow involved in a stochastic
    # operation.

    # During graph accumulation we accumulate untyped constant nodes for
    # all non-stochastic values involved in a stochastic operation regardless
    # of whether or not they can be represented in BMG.  During a later
    # pass we give error messages if we are unable to replace the unsupported
    # constant values with valid BMG nodes.

    @memoize
    def _add_constant(self, value: Any, t: type) -> bn.UntypedConstantNode:
        # Note that we memoize *after* we've canonicalized the value,
        # and we ensure that the type is part of the memoization key.
        # We do not want to get into a situation where some unexpected Python
        # rule says that the constant 1 and the constant 1.0 are the same.
        node = bn.UntypedConstantNode(value)
        self.add_node(node)
        return node

    def add_constant(self, value: Any) -> bn.UntypedConstantNode:
        """This takes any constant value of a supported type,
        creates a constant graph node for it, and adds it to the builder"""
        t = type(value)
        if t in supported_bool_types:
            value = bool(value)
            t = bool
        elif t in supported_int_types:
            value = int(value)
            t = int
        elif t in supported_float_types:
            value = float(value)
            t = float
        return self._add_constant(value, t)

    def add_constant_of_matrix_type(
        self, value: Any, node_type: bt.BMGMatrixType
    ) -> ConstantNode:
        # If we need a simplex, add a simplex. Otherwise,
        # choose which kind of matrix node to create based on
        # the matrix element type.
        if isinstance(node_type, bt.SimplexMatrix):
            return self.add_simplex(value)
        if node_type.element_type == bt.real_element:
            return self.add_real_matrix(value)
        if node_type.element_type == bt.positive_real_element:
            return self.add_pos_real_matrix(value)
        if node_type.element_type == bt.negative_real_element:
            return self.add_neg_real_matrix(value)
        if node_type.element_type == bt.probability_element:
            return self.add_probability_matrix(value)
        if node_type.element_type == bt.natural_element:
            return self.add_natural_matrix(value)
        if node_type.element_type == bt.bool_element:
            return self.add_boolean_matrix(value)
        raise NotImplementedError(
            "add_constant_of_matrix_type not yet "
            + f"implemented for {node_type.long_name}"
        )

    def add_constant_of_type(
        self, value: Any, node_type: bt.BMGLatticeType
    ) -> ConstantNode:
        """This takes any constant value of a supported type and creates a
        constant graph node of the stated type for it, and adds it to the builder"""
        if node_type == bt.Boolean:
            return self.add_boolean(bool(value))
        if node_type == bt.Probability:
            return self.add_probability(float(value))
        if node_type == bt.Natural:
            return self.add_natural(int(value))
        if node_type == bt.PositiveReal:
            return self.add_pos_real(float(value))
        if node_type == bt.NegativeReal:
            return self.add_neg_real(float(value))
        if node_type == bt.Real:
            return self.add_real(float(value))
        if node_type == bt.Tensor:
            if isinstance(value, torch.Tensor):
                return self.add_constant_tensor(value)
            return self.add_constant_tensor(torch.tensor(value))
        if isinstance(node_type, bt.BMGMatrixType):
            return self.add_constant_of_matrix_type(value, node_type)
        raise NotImplementedError(
            "add_constant_of_type not yet " + f"implemented for {node_type.long_name}"
        )

    @memoize
    def add_real(self, value: float) -> bn.RealNode:
        node = bn.RealNode(value)
        self.add_node(node)
        return node

    @memoize
    def add_probability(self, value: float) -> bn.ProbabilityNode:
        node = bn.ProbabilityNode(value)
        self.add_node(node)
        return node

    @memoize
    def add_pos_real(self, value: float) -> bn.PositiveRealNode:
        node = bn.PositiveRealNode(value)
        self.add_node(node)
        return node

    @memoize
    def add_boolean_matrix(self, value: torch.Tensor) -> bn.ConstantBooleanMatrixNode:
        assert len(value.size()) <= 2
        node = bn.ConstantBooleanMatrixNode(value)
        self.add_node(node)
        return node

    @memoize
    def add_natural_matrix(self, value: torch.Tensor) -> bn.ConstantNaturalMatrixNode:
        assert len(value.size()) <= 2
        node = bn.ConstantNaturalMatrixNode(value)
        self.add_node(node)
        return node

    @memoize
    def add_probability_matrix(
        self, value: torch.Tensor
    ) -> bn.ConstantProbabilityMatrixNode:
        assert len(value.size()) <= 2
        node = bn.ConstantProbabilityMatrixNode(value)
        self.add_node(node)
        return node

    @memoize
    def add_simplex(self, value: torch.Tensor) -> bn.ConstantSimplexMatrixNode:
        assert len(value.size()) <= 2
        node = bn.ConstantSimplexMatrixNode(value)
        self.add_node(node)
        return node

    @memoize
    def add_pos_real_matrix(
        self, value: torch.Tensor
    ) -> bn.ConstantPositiveRealMatrixNode:
        assert len(value.size()) <= 2
        node = bn.ConstantPositiveRealMatrixNode(value)
        self.add_node(node)
        return node

    @memoize
    def add_neg_real_matrix(
        self, value: torch.Tensor
    ) -> bn.ConstantNegativeRealMatrixNode:
        assert len(value.size()) <= 2
        node = bn.ConstantNegativeRealMatrixNode(value)
        self.add_node(node)
        return node

    @memoize
    def add_real_matrix(self, value: torch.Tensor) -> bn.ConstantRealMatrixNode:
        assert len(value.size()) <= 2
        node = bn.ConstantRealMatrixNode(value)
        self.add_node(node)
        return node

    @memoize
    def add_neg_real(self, value: float) -> bn.NegativeRealNode:
        node = bn.NegativeRealNode(value)
        self.add_node(node)
        return node

    @memoize
    def add_natural(self, value: int) -> bn.NaturalNode:
        node = bn.NaturalNode(value)
        self.add_node(node)
        return node

    @memoize
    def add_boolean(self, value: bool) -> bn.BooleanNode:
        node = bn.BooleanNode(value)
        self.add_node(node)
        return node

    @memoize
    def add_constant_tensor(self, value: torch.Tensor) -> bn.ConstantTensorNode:
        node = bn.ConstantTensorNode(value)
        self.add_node(node)
        return node

    # ####
    # #### Graph accumulation for distributions
    # ####

    # TODO: This code is mostly but not entirely in alpha order
    # by distribution type; we might reorganize it to make it
    # slightly easier to follow.

    @memoize
    def add_bernoulli(self, probability: BMGNode) -> bn.BernoulliNode:
        node = bn.BernoulliNode(probability)
        self.add_node(node)
        return node

    @memoize
    def add_bernoulli_logit(self, probability: BMGNode) -> bn.BernoulliLogitNode:
        node = bn.BernoulliLogitNode(probability)
        self.add_node(node)
        return node

    @memoize
    def add_binomial(self, count: BMGNode, probability: BMGNode) -> bn.BinomialNode:
        node = bn.BinomialNode(count, probability)
        self.add_node(node)
        return node

    @memoize
    def add_binomial_logit(
        self, count: BMGNode, probability: BMGNode
    ) -> bn.BinomialLogitNode:
        node = bn.BinomialLogitNode(count, probability)
        self.add_node(node)
        return node

    @memoize
    def add_categorical(self, probability: BMGNode) -> bn.CategoricalNode:
        node = bn.CategoricalNode(probability)
        self.add_node(node)
        return node

    @memoize
    def add_categorical_logit(self, probability: BMGNode) -> bn.CategoricalLogitNode:
        node = bn.CategoricalLogitNode(probability)
        self.add_node(node)
        return node

    @memoize
    def add_chi2(self, df: BMGNode) -> bn.Chi2Node:
        node = bn.Chi2Node(df)
        self.add_node(node)
        return node

    @memoize
    def add_gamma(self, concentration: BMGNode, rate: BMGNode) -> bn.GammaNode:
        node = bn.GammaNode(concentration, rate)
        self.add_node(node)
        return node

    @memoize
    def add_halfcauchy(self, scale: BMGNode) -> bn.HalfCauchyNode:
        node = bn.HalfCauchyNode(scale)
        self.add_node(node)
        return node

    @memoize
    def add_normal(self, mu: BMGNode, sigma: BMGNode) -> bn.NormalNode:
        node = bn.NormalNode(mu, sigma)
        self.add_node(node)
        return node

    @memoize
    def add_halfnormal(self, sigma: BMGNode) -> bn.HalfNormalNode:
        node = bn.HalfNormalNode(sigma)
        self.add_node(node)
        return node

    @memoize
    def add_dirichlet(self, concentration: BMGNode) -> bn.DirichletNode:
        node = bn.DirichletNode(concentration)
        self.add_node(node)
        return node

    @memoize
    def add_studentt(
        self, df: BMGNode, loc: BMGNode, scale: BMGNode
    ) -> bn.StudentTNode:
        node = bn.StudentTNode(df, loc, scale)
        self.add_node(node)
        return node

    @memoize
    def add_uniform(self, low: BMGNode, high: BMGNode) -> bn.UniformNode:
        node = bn.UniformNode(low, high)
        self.add_node(node)
        return node

    @memoize
    def add_beta(self, alpha: BMGNode, beta: BMGNode) -> bn.BetaNode:
        node = bn.BetaNode(alpha, beta)
        self.add_node(node)
        return node

    @memoize
    def add_poisson(self, rate: BMGNode) -> bn.PoissonNode:
        node = bn.PoissonNode(rate)
        self.add_node(node)
        return node

    @memoize
    def add_flat(self) -> bn.FlatNode:
        node = bn.FlatNode()
        self.add_node(node)
        return node

    # ####
    # #### Graph accumulation for operators
    # ####

    # The handler methods here are both invoked directly, when, say
    # there was an explicit addition in the original model, and
    # indirectly as the result of processing a function call such
    # as tensor.add.

    # TODO: This code is not very well organized; consider sorting it
    # into alpha order by operation.

    @memoize
    def add_greater_than(self, left: BMGNode, right: BMGNode) -> BMGNode:
        if isinstance(left, ConstantNode):
            if isinstance(right, ConstantNode):
                return self.add_constant(left.value > right.value)

        node = bn.GreaterThanNode(left, right)
        self.add_node(node)
        return node

    @memoize
    def add_greater_than_equal(self, left: BMGNode, right: BMGNode) -> BMGNode:
        if isinstance(left, ConstantNode):
            if isinstance(right, ConstantNode):
                return self.add_constant(left.value >= right.value)

        node = bn.GreaterThanEqualNode(left, right)
        self.add_node(node)
        return node

    @memoize
    def add_less_than(self, left: BMGNode, right: BMGNode) -> BMGNode:
        if isinstance(left, ConstantNode):
            if isinstance(right, ConstantNode):
                return self.add_constant(left.value < right.value)

        node = bn.LessThanNode(left, right)
        self.add_node(node)
        return node

    @memoize
    def add_less_than_equal(self, left: BMGNode, right: BMGNode) -> BMGNode:
        if isinstance(left, ConstantNode):
            if isinstance(right, ConstantNode):
                return self.add_constant(left.value <= right.value)

        node = bn.LessThanEqualNode(left, right)
        self.add_node(node)
        return node

    @memoize
    def add_equal(self, left: BMGNode, right: BMGNode) -> BMGNode:
        if isinstance(left, ConstantNode):
            if isinstance(right, ConstantNode):
                return self.add_constant(left.value == right.value)

        node = bn.EqualNode(left, right)
        self.add_node(node)
        return node

    @memoize
    def add_not_equal(self, left: BMGNode, right: BMGNode) -> BMGNode:
        if isinstance(left, ConstantNode):
            if isinstance(right, ConstantNode):
                return self.add_constant(left.value != right.value)

        node = bn.NotEqualNode(left, right)
        self.add_node(node)
        return node

    @memoize
    def add_is(self, left: BMGNode, right: BMGNode) -> BMGNode:
        node = bn.IsNode(left, right)
        self.add_node(node)
        return node

    @memoize
    def add_is_not(self, left: BMGNode, right: BMGNode) -> BMGNode:
        node = bn.IsNotNode(left, right)
        self.add_node(node)
        return node

    @memoize
    def add_in(self, left: BMGNode, right: BMGNode) -> BMGNode:
        node = bn.InNode(left, right)
        self.add_node(node)
        return node

    @memoize
    def add_not_in(self, left: BMGNode, right: BMGNode) -> BMGNode:
        node = bn.NotInNode(left, right)
        self.add_node(node)
        return node

    @memoize
    def add_addition(self, left: BMGNode, right: BMGNode) -> BMGNode:
        if isinstance(left, ConstantNode) and isinstance(right, ConstantNode):
            return self.add_constant(left.value + right.value)

        inps = [left, right]
        node = bn.AdditionNode(inps)
        self.add_node(node)
        return node

    @memoize
    def add_bitand(self, left: BMGNode, right: BMGNode) -> BMGNode:
        node = bn.BitAndNode(left, right)
        self.add_node(node)
        return node

    @memoize
    def add_bitor(self, left: BMGNode, right: BMGNode) -> BMGNode:
        node = bn.BitOrNode(left, right)
        self.add_node(node)
        return node

    @memoize
    def add_bitxor(self, left: BMGNode, right: BMGNode) -> BMGNode:
        node = bn.BitXorNode(left, right)
        self.add_node(node)
        return node

    @memoize
    def add_floordiv(self, left: BMGNode, right: BMGNode) -> BMGNode:
        node = bn.FloorDivNode(left, right)
        self.add_node(node)
        return node

    @memoize
    def add_lshift(self, left: BMGNode, right: BMGNode) -> BMGNode:
        node = bn.LShiftNode(left, right)
        self.add_node(node)
        return node

    @memoize
    def add_mod(self, left: BMGNode, right: BMGNode) -> BMGNode:
        node = bn.ModNode(left, right)
        self.add_node(node)
        return node

    @memoize
    def add_rshift(self, left: BMGNode, right: BMGNode) -> BMGNode:
        node = bn.RShiftNode(left, right)
        self.add_node(node)
        return node

    # No need to memoize this since the addition will be memoized.
    def add_subtraction(self, left: BMGNode, right: BMGNode) -> BMGNode:
        # TODO: We don't have a subtraction node; we render this as
        # left + (-right), which we do have.  Should we have a subtraction
        # node? We could do this transformation in a problem-fixing pass,
        # like we do for division.
        return self.add_addition(left, self.add_negate(right))

    @memoize
    def add_multi_addition(self, *inputs: BMGNode) -> BMGNode:
        node = bn.AdditionNode(list(inputs))
        self.add_node(node)
        return node

    @memoize
    def add_multiplication(self, left: BMGNode, right: BMGNode) -> BMGNode:
        if isinstance(left, ConstantNode) and isinstance(right, ConstantNode):
            return self.add_constant(left.value * right.value)
        inps = [left, right]
        node = bn.MultiplicationNode(inps)
        self.add_node(node)
        return node

    @memoize
    def add_multi_multiplication(self, *inputs: BMGNode) -> BMGNode:
        node = bn.MultiplicationNode(list(inputs))
        self.add_node(node)
        return node

    @memoize
    def add_if_then_else(
        self, condition: BMGNode, consequence: BMGNode, alternative: BMGNode
    ) -> BMGNode:
        # If the condition is a constant then we can optimize away the if-then-else
        # node entirely.
        if bn.is_one(condition):
            return consequence
        if bn.is_zero(condition):
            return alternative
        node = bn.IfThenElseNode(condition, consequence, alternative)
        self.add_node(node)
        return node

    @memoize
    def add_choice(self, condition: BMGNode, *values: BMGNode) -> bn.ChoiceNode:
        vs = list(values)
        assert len(values) >= 2
        node = bn.ChoiceNode(condition, vs)
        self.add_node(node)
        return node

    @memoize
    def add_matrix_multiplication(self, left: BMGNode, right: BMGNode) -> BMGNode:
        if isinstance(left, ConstantNode) and isinstance(right, ConstantNode):
            return self.add_constant(torch.mm(left.value, right.value))
        node = bn.MatrixMultiplicationNode(left, right)
        self.add_node(node)
        return node

    @memoize
    def add_matrix_scale(self, scalar: BMGNode, matrix: BMGNode) -> BMGNode:
        # Intended convention here is that the scalar comes first.
        # However, this cannot be checked here
        # TODO[Walid]: Fix to match reverse order convention of torch.mul
        if isinstance(scalar, ConstantNode) and isinstance(matrix, ConstantNode):
            return self.add_constant(scalar.value * matrix.value)
        node = bn.MatrixScaleNode(scalar, matrix)
        self.add_node(node)
        return node

    @memoize
    def add_division(self, left: BMGNode, right: BMGNode) -> BMGNode:
        if isinstance(left, ConstantNode) and isinstance(right, ConstantNode):
            return self.add_constant(left.value / right.value)
        node = bn.DivisionNode(left, right)
        self.add_node(node)
        return node

    @memoize
    def add_power(self, left: BMGNode, right: BMGNode) -> BMGNode:
        if isinstance(left, ConstantNode) and isinstance(right, ConstantNode):
            return self.add_constant(left.value ** right.value)
        node = bn.PowerNode(left, right)
        self.add_node(node)
        return node

    @memoize
    def add_index(self, left: bn.BMGNode, right: bn.BMGNode) -> bn.BMGNode:
        # Folding optimizations are done in the fixer.
        node = bn.IndexNode(left, right)
        self.add_node(node)
        return node

    @memoize
    def add_item(self, input: bn.BMGNode) -> bn.BMGNode:
        node = bn.ItemNode(input)
        self.add_node(node)
        return node

    @memoize
    def add_vector_index(self, left: bn.BMGNode, right: bn.BMGNode) -> bn.BMGNode:
        # Folding optimizations are done in the fixer.
        node = bn.VectorIndexNode(left, right)
        self.add_node(node)
        return node

    @memoize
    def add_column_index(self, left: bn.BMGNode, right: bn.BMGNode) -> bn.BMGNode:
        # Folding optimizations are done in the fixer.
        node = bn.ColumnIndexNode(left, right)
        self.add_node(node)
        return node

    @memoize
    def add_negate(self, operand: BMGNode) -> BMGNode:
        # TODO: We could optimize -(-x) to x here.
        if isinstance(operand, ConstantNode):
            return self.add_constant(-operand.value)
        node = bn.NegateNode(operand)
        self.add_node(node)
        return node

    @memoize
    def add_invert(self, operand: BMGNode) -> BMGNode:
        node = bn.InvertNode(operand)
        self.add_node(node)
        return node

    @memoize
    def add_complement(self, operand: BMGNode) -> BMGNode:
        if isinstance(operand, ConstantNode):
            return self.add_constant(1 - operand.value)
        node = bn.ComplementNode(operand)
        self.add_node(node)
        return node

    # TODO: What should the result of NOT on a tensor be?
    # TODO: Should it be legal at all in the graph?
    # TODO: In Python, (not tensor(x)) is equal to (not x).
    # TODO: It is NOT equal to (tensor(not x)), which is what
    # TODO: you might expect.
    @memoize
    def add_not(self, operand: BMGNode) -> BMGNode:
        if isinstance(operand, ConstantNode):
            return self.add_constant(not operand.value)
        node = bn.NotNode(operand)
        self.add_node(node)
        return node

    @memoize
    def add_to_real(self, operand: BMGNode) -> BMGNode:
        if isinstance(operand, bn.RealNode):
            return operand
        if isinstance(operand, ConstantNode):
            return self.add_real(float(operand.value))
        node = bn.ToRealNode(operand)
        self.add_node(node)
        return node

    @memoize
    def add_to_int(self, operand: BMGNode) -> BMGNode:
        if isinstance(operand, bn.ConstantTensorNode):
            return self.add_constant(operand.value.int)
        if isinstance(operand, ConstantNode):
            return self.add_constant(int(operand.value))
        node = bn.ToIntNode(operand)
        self.add_node(node)
        return node

    @memoize
    def add_to_real_matrix(self, operand: BMGNode) -> BMGNode:
        node = bn.ToRealMatrixNode(operand)
        self.add_node(node)
        return node

    @memoize
    def add_to_positive_real(self, operand: BMGNode) -> BMGNode:
        if isinstance(operand, bn.PositiveRealNode):
            return operand
        if isinstance(operand, ConstantNode):
            return self.add_to_positive_real(float(operand.value))
        node = bn.ToPositiveRealNode(operand)
        self.add_node(node)
        return node

    @memoize
    def add_to_positive_real_matrix(self, operand: BMGNode) -> BMGNode:
        node = bn.ToPositiveRealMatrixNode(operand)
        self.add_node(node)
        return node

    @memoize
    def add_to_probability(self, operand: BMGNode) -> BMGNode:
        if isinstance(operand, bn.ProbabilityNode):
            return operand
        if isinstance(operand, ConstantNode):
            return self.add_probability(float(operand.value))
        node = bn.ToProbabilityNode(operand)
        self.add_node(node)
        return node

    @memoize
    def add_to_negative_real(self, operand: BMGNode) -> BMGNode:
        node = bn.ToNegativeRealNode(operand)
        self.add_node(node)
        return node

    @memoize
    def add_exp(self, operand: BMGNode) -> BMGNode:
        if isinstance(operand, bn.ConstantTensorNode):
            return self.add_constant(torch.exp(operand.value))
        if isinstance(operand, ConstantNode):
            return self.add_constant(math.exp(operand.value))
        node = bn.ExpNode(operand)
        self.add_node(node)
        return node

    @memoize
    def add_expm1(self, operand: BMGNode) -> BMGNode:
        if isinstance(operand, bn.ConstantTensorNode):
            return self.add_constant(torch.expm1(operand.value))
        if isinstance(operand, ConstantNode):
            return self.add_constant(torch.expm1(torch.tensor(operand.value)))
        node = bn.ExpM1Node(operand)
        self.add_node(node)
        return node

    @memoize
    def add_logistic(self, operand: BMGNode) -> BMGNode:
        if isinstance(operand, bn.ConstantTensorNode):
            return self.add_constant(torch.sigmoid(operand.value))
        if isinstance(operand, ConstantNode):
            return self.add_constant(torch.sigmoid(torch.tensor(operand.value)))
        node = bn.LogisticNode(operand)
        self.add_node(node)
        return node

    @memoize
    def add_phi(self, operand: BMGNode) -> BMGNode:
        if isinstance(operand, ConstantNode):
            return self.add_constant(phi(operand.value))
        node = bn.PhiNode(operand)
        self.add_node(node)
        return node

    @memoize
    def add_log(self, operand: BMGNode) -> BMGNode:
        if isinstance(operand, bn.ConstantTensorNode):
            return self.add_constant(torch.log(operand.value))
        if isinstance(operand, ConstantNode):
            return self.add_constant(math.log(operand.value))
        node = bn.LogNode(operand)
        self.add_node(node)
        return node

    @memoize
    def add_log1mexp(self, operand: BMGNode) -> BMGNode:
        if isinstance(operand, bn.ConstantTensorNode):
            return self.add_constant(log1mexp(operand.value))
        if isinstance(operand, ConstantNode):
            return self.add_constant(math_log1mexp(operand.value))
        node = bn.Log1mexpNode(operand)
        self.add_node(node)
        return node

    @memoize
    def add_tensor(self, size: torch.Size, *data: BMGNode) -> bn.TensorNode:
        node = bn.TensorNode(list(data), size)
        self.add_node(node)
        return node

    @memoize
    def add_to_matrix(
        self, rows: bn.NaturalNode, columns: bn.NaturalNode, *data: BMGNode
    ) -> bn.ToMatrixNode:
        node = bn.ToMatrixNode(rows, columns, list(data))
        self.add_node(node)
        return node

    @memoize
    def add_logsumexp(self, *inputs: BMGNode) -> bn.LogSumExpNode:
        node = bn.LogSumExpNode(list(inputs))
        self.add_node(node)
        return node

    @memoize
    def add_logsumexp_torch(
        self, input: BMGNode, dim: BMGNode, keepdim: BMGNode
    ) -> bn.LogSumExpTorchNode:
        node = bn.LogSumExpTorchNode(input, dim, keepdim)
        self.add_node(node)
        return node

    @memoize
    def add_logsumexp_vector(self, operand: BMGNode) -> bn.LogSumExpVectorNode:
        node = bn.LogSumExpVectorNode(operand)
        self.add_node(node)
        return node

    @memoize
    def add_switch(self, *elements: BMGNode) -> bn.SwitchNode:
        # TODO: Verify that the list is well-formed.
        node = bn.SwitchNode(list(elements))
        self.add_node(node)
        return node

    # Do NOT memoize add_sample; each sample node must be unique
    def add_sample(self, operand: bn.DistributionNode) -> bn.SampleNode:
        node = bn.SampleNode(operand)
        self.add_node(node)
        return node

    # TODO: Should this be idempotent?
    # TODO: Should it be an error to add two unequal observations to one node?
    def add_observation(self, observed: bn.SampleNode, value: Any) -> bn.Observation:
        node = bn.Observation(observed, value)
        self.add_node(node)
        return node

    @memoize
    def add_query(self, operator: BMGNode) -> bn.Query:
        # TODO: BMG requires that the target of a query be classified
        # as an operator and that queries be unique; that is, every node
        # is queried *exactly* zero or one times. Rather than making
        # those restrictions here, instead detect bad queries in the
        # problem fixing phase and report accordingly.
        node = bn.Query(operator)
        self.add_node(node)
        return node

    def add_exp_product(self, *inputs: BMGNode) -> bn.ExpProductFactorNode:
        # Note that factors are NOT deduplicated; this method is not
        # memoized. We need to be able to add multiple factors to the same
        # node, similar to the way we need to add multiple samples to a
        # distribution.
        node = bn.ExpProductFactorNode(list(inputs))
        self.add_node(node)
        return node

    def all_ancestor_nodes(self) -> List[BMGNode]:
        """Returns a topo-sorted list of nodes that are ancestors to any
        sample, observation, query or factor."""

        def is_root(n: BMGNode) -> bool:
            return (
                isinstance(n, bn.SampleNode)
                or isinstance(n, bn.Observation)
                or isinstance(n, bn.Query)
                or isinstance(n, bn.FactorNode)
            )

        return self._traverse(is_root)

    def all_nodes(self) -> List[BMGNode]:
        """Returns a topo-sorted list of all nodes."""
        return self._traverse(lambda n: n.is_leaf)

    def _traverse(self, is_root: Callable[[BMGNode], bool]) -> List[BMGNode]:
        """This returns a list of the reachable graph nodes
        in topologically sorted order. The ordering invariants are
        (1) all sample, observation, query and factor nodes are
        enumerated in the order they were added, and
        (2) all inputs are enumerated before their outputs, and
        (3) inputs to the "left" are enumerated before those to
        the "right"."""

        # We require here that the graph is acyclic.

        # TODO: The graph should be acyclic by construction;
        # we detect cycles while executing the lifted model.
        # However, we might want to add a quick cycle checking
        # pass here as a sanity check.

        def key(n: BMGNode) -> int:
            return self._nodes[n]

        # We cannot use a recursive algorithm because the graph may have
        # paths that are deeper than the recursion limit in Python.
        # Instead we'll use a list as a stack.  But we cannot simply do
        # a normal iterative depth-first or postorder traversal because
        # that violates our stated invariants above: all inputs are always
        # enumerated before the node which inputs them, and nodes to the
        # left are enumerated before nodes to the right.
        #
        # What we do here is a modified depth first traversal which maintains
        # our invariants.

        result = []
        work_stack = sorted(
            (n for n in self._nodes if is_root(n)), key=key, reverse=True
        )
        already_in_result = set()
        inputs_already_pushed = set()

        while len(work_stack) != 0:
            # Peek the top of the stack but do not pop it yet.
            current = work_stack[-1]
            if current in already_in_result:
                # The top of the stack has already been put into the
                # result list. There is nothing more to do with this node,
                # so we can simply pop it away.
                work_stack.pop()
            elif current in inputs_already_pushed:
                # The top of the stack is not on the result list, but we have
                # already pushed all of its inputs onto the stack. Since they
                # are gone from the stack, we must have already put all of them
                # onto the result list, and therefore we are justified in putting
                # this node onto the result list too.
                work_stack.pop()
                result.append(current)
                already_in_result.add(current)
            else:
                # The top of the stack is not on the result list and its inputs
                # have never been put onto the stack. Leave it on the stack so that
                # we come back to it later after all of its inputs have been
                # put on the result list, and put its inputs on the stack.
                #
                # We want to process the left inputs before the right inputs, so
                # reverse them so that the left inputs go on the stack last, and
                # are therefore closer to the top.
                for i in reversed(current.inputs):
                    work_stack.append(i)
                inputs_already_pushed.add(current)

        return result

    def all_observations(self) -> List[bn.Observation]:
        return sorted(
            (n for n in self._nodes if isinstance(n, bn.Observation)),
            key=lambda n: self._nodes[n],
        )
