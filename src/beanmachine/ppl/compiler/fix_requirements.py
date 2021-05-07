# Copyright (c) Facebook, Inc. and its affiliates.

"""This module takes a Bean Machine Graph builder and makes a best
effort attempt to transform the accumulated graph to meet the
requirements of the BMG type system. All possible transformations
are made; if there are nodes that cannot be represented in BMG
or cannot be made to meet type requirements, an error report is
returned."""


import beanmachine.ppl.compiler.bmg_nodes as bn
import beanmachine.ppl.compiler.bmg_types as bt
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_requirements import EdgeRequirements
from beanmachine.ppl.compiler.error_report import ErrorReport, Violation
from beanmachine.ppl.compiler.graph_labels import get_edge_labels
from beanmachine.ppl.compiler.lattice_typer import LatticeTyper
from torch import Tensor


class RequirementsFixer:
    """This class takes a Bean Machine Graph builder and attempts to
    fix violations of BMG type system requirements.

    The basic idea is that every *edge* in the graph has a *requirement*, such as
    "the type of the input must be Probability".  We do a traversal of the input
    edges of every node in the graph; if the input node meets the requirement,
    it is unchanged. If it does not, then a new node that has the same semantics
    that meets the requirement is returned. If there is no such node then an
    error is added to the error report."""

    errors: ErrorReport
    bmg: BMGraphBuilder
    _typer: LatticeTyper
    _reqs: EdgeRequirements

    def __init__(self, bmg: BMGraphBuilder, typer: LatticeTyper) -> None:
        self.errors = ErrorReport()
        self.bmg = bmg
        self._typer = typer
        self._reqs = EdgeRequirements(typer)

    def _type_meets_requirement(self, t: bt.BMGLatticeType, r: bt.Requirement) -> bool:
        assert t != bt.Untypable
        if isinstance(r, bt.AnyRequirement):
            return True
        if isinstance(r, bt.UpperBound):
            return bt.supremum(t, r.bound) == r.bound
        if isinstance(r, bt.AlwaysMatrix):
            return t == r.bound
        return t == r

    def _node_meets_requirement(self, node: bn.BMGNode, r: bt.Requirement) -> bool:
        if isinstance(r, bt.AlwaysMatrix):
            return node.is_matrix and self._type_meets_requirement(
                self._typer.graph_type(node), r.bound
            )
        return self._type_meets_requirement(self._typer.graph_type(node), r)

    def _meet_constant_requirement(
        self,
        node: bn.ConstantNode,
        requirement: bt.Requirement,
        consumer: bn.BMGNode,
        edge: str,
    ) -> bn.BMGNode:
        # If the constant node already meets the requirement, we're done.
        if not isinstance(
            node, bn.UntypedConstantNode
        ) and self._node_meets_requirement(node, requirement):
            return node

        # It does not meet the requirement. Is there a semantically equivalent node
        # that does meet the requirement?

        # The inf type is defined as the smallest type to which the node can be converted.
        # If the infimum type is smaller than or equal to the required type, then the
        # node can definitely be converted to a type which meets the requirement.
        it = self._typer[node]
        if self._type_meets_requirement(it, bt.upper_bound(requirement)):

            # To what type should we convert the node to meet the requirement?
            # If the requirement is an exact bound, then that's the type we need to
            # convert to. If the requirement is an upper bound, there's no reason
            # why we can't just convert to that type.

            required_type = bt.requirement_to_type(requirement)
            if bt.must_be_matrix(requirement):
                assert isinstance(required_type, bt.BMGMatrixType)
                result = self.bmg.add_constant_of_matrix_type(node.value, required_type)
            else:
                result = self.bmg.add_constant_of_type(node.value, required_type)
            assert self._node_meets_requirement(result, requirement)
            return result

        # We cannot convert this node to any type that meets the requirement.
        # Add an error.
        self.errors.add_error(Violation(node, it, requirement, consumer, edge))
        return node

    def _meet_distribution_requirement(
        self,
        node: bn.DistributionNode,
        requirement: bt.Requirement,
        consumer: bn.BMGNode,
        edge: str,
    ) -> bn.BMGNode:
        # The only edges which point to distributions are samples, and the requirement
        # on that edge is always met automatically.
        assert isinstance(consumer, bn.SampleNode)
        assert requirement == self._typer[node]
        return node

    def _convert_node(
        self,
        node: bn.OperatorNode,
        requirement: bt.BMGLatticeType,
        consumer: bn.BMGNode,
        edge: str,
    ) -> bn.BMGNode:
        # We have been given a node which does not meet a requirement,
        # but it can be converted to a node which does meet the requirement
        # that has the same semantics. Start by confirming those preconditions.
        assert self._typer.graph_type(node) != requirement
        assert bt.supremum(self._typer[node], requirement) == requirement

        # TODO: We no longer support Tensor as a type in BMG.  We must
        # detect, and produce a good error message, for situations
        # where we have deduced that the only possible type of a node is
        # a >2-dimension tensor; we must correctly support cases where
        # the type of the node is a 1- or 2-dimensional tensor.

        if requirement == Tensor:
            raise ValueError("Unsupported type requirement: Tensor")

        # Converting anything to real or positive real is easy;
        # there's already a node for that so just insert it on the edge
        # whose requirement is not met, and the requirement will be met.

        if requirement == bt.Real:
            return self.bmg.add_to_real(node)
        if requirement == bt.PositiveReal:
            return self.bmg.add_to_positive_real(node)

        # We are not converting to real or positive real.
        # Our precondition is that the requirement is larger than
        # *something*, which means that it cannot be bool.
        # That means the requirement must be either natural or
        # probability. Verify this.

        assert requirement == bt.Natural or requirement == bt.Probability

        # Our precondition is that the requirement is larger than the
        # inf type of the node.

        assert self._typer.is_bool(node)

        # There is no "to natural" or "to probability" but since we have
        # a bool in hand, we can use an if-then-else as a conversion.

        zero = self.bmg.add_constant_of_type(0.0, requirement)
        one = self.bmg.add_constant_of_type(1.0, requirement)
        return self.bmg.add_if_then_else(node, one, zero)

    def _can_force_to_prob(
        self, inf_type: bt.BMGLatticeType, requirement: bt.Requirement
    ) -> bool:
        # Consider the graph created by a call like:
        #
        # Bernoulli(0.5 + some_beta() / 2)
        #
        # The inf types of the addends are both probability, but there is
        # no addition operator on probabilities; we will add these as
        # positive reals, and then get an error when we use it as the parameter
        # to a Bernoulli.  But you and I both know that this is a legal
        # probability.
        #
        # To work around this problem, if we have a *real* or *positive real* used
        # in a situation where a *probability* is required, we insert an explicit
        # "clamp this real to a probability" operation.
        #
        # TODO: We might want to restrict this. For example, if we have
        #
        # Bernoulli(some_normal())
        #
        # then it seems plausible that we ought to produce an error here rather than
        # clamping the result to a probability. We could allow this feature only
        # in situations where there was some operator other than a sample, for instance.
        #
        # TODO: We might want to build a warning mechanism that informs the developer
        # of the possibility that they've gotten something wrong here.
        return (
            requirement == bt.Probability
            or requirement == bt.upper_bound(bt.Probability)
        ) and (inf_type == bt.Real or inf_type == bt.PositiveReal)

    def _meet_operator_requirement(
        self,
        node: bn.OperatorNode,
        requirement: bt.Requirement,
        consumer: bn.BMGNode,
        edge: str,
    ) -> bn.BMGNode:
        # If the operator node already meets the requirement, we're done.
        if self._node_meets_requirement(node, requirement):
            return node

        # It does not meet the requirement. Can we convert this thing to a node
        # whose type does meet the requirement? Remember, the inf type is the
        # smallest type that this node is convertible to, so if the inf type
        # meets an upper bound requirement, then the conversion we want exists.

        it = self._typer[node]

        if not self._type_meets_requirement(it, bt.upper_bound(requirement)):
            # We cannot make the node meet the requirement "implicitly". However
            # there is one situation where we can "explicitly" meet a requirement:
            # an operator of type real or positive real used as a probability.
            if self._can_force_to_prob(it, requirement):
                # Ensure that the operand is converted to real or positive real:
                operand = self.meet_requirement(node, it, consumer, edge)
                # Force the real / positive real to probability:
                result = self.bmg.add_to_probability(operand)
                assert self._node_meets_requirement(result, requirement)
                return result

            # We have no way to make the conversion we need, so add an error.
            self.errors.add_error(Violation(node, it, requirement, consumer, edge))
            return node

        # We definitely can meet the requirement; it just remains to figure
        # out exactly how.
        #
        # There are now two possibilities:
        #
        # * The requirement is an exact requirement. We know that the node
        #   can be converted to that type, because its inf type meets an
        #   upper bound requirement. Convert it to that exact type.
        #
        # * The requirement is an upper-bound requirement, and the inf type
        #   meets it. Convert the node to the inf type.

        if isinstance(requirement, bt.BMGLatticeType):
            result = self._convert_node(node, requirement, consumer, edge)
        else:
            result = self._convert_node(node, it, consumer, edge)

        # TODO: This assertion could fire if we require a positive real matrix
        # but the result of the conversion is a positive real value.  We need
        # to handle that case.

        assert self._node_meets_requirement(result, requirement)
        return result

    def meet_requirement(
        self,
        node: bn.BMGNode,
        requirement: bt.Requirement,
        consumer: bn.BMGNode,
        edge: str,
    ) -> bn.BMGNode:
        """The consumer node consumes the value of the input node. The consumer's
        requirement is given; the name of this edge is provided for error reporting."""

        if isinstance(node, bn.Observation):
            raise AssertionError(
                "Unexpected graph topology; an observation is never an input"
            )
        if isinstance(node, bn.Query):
            raise AssertionError("Unexpected graph topology; a query is never an input")
        if isinstance(node, bn.ConstantNode):
            return self._meet_constant_requirement(node, requirement, consumer, edge)
        if isinstance(node, bn.DistributionNode):
            return self._meet_distribution_requirement(
                node, requirement, consumer, edge
            )
        if isinstance(node, bn.OperatorNode):
            return self._meet_operator_requirement(node, requirement, consumer, edge)
        raise AssertionError("Unexpected node type")

    def fix_problems(self) -> None:
        nodes = self.bmg.all_ancestor_nodes()
        for node in nodes:
            requirements = self._reqs.requirements(node)
            # TODO: The edge labels used to visualize the graph in DOT
            # are not necessarily the best ones for displaying errors.
            # Consider fixing this.
            edges = get_edge_labels(node)
            node_was_updated = False
            for i in range(len(requirements)):
                new_input = self.meet_requirement(
                    node.inputs[i], requirements[i], node, edges[i]
                )
                if node.inputs[i] is not new_input:
                    node.inputs[i] = new_input
                    node_was_updated = True
            if node_was_updated:
                self._typer.update_type(node)
