# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""This module takes a Bean Machine Graph builder and makes a best
effort attempt to transform the accumulated graph to meet the
requirements of the BMG type system. All possible transformations
are made; if there are nodes that cannot be represented in BMG
or cannot be made to meet type requirements, an error report is
returned."""


from typing import Optional

import beanmachine.ppl.compiler.bmg_nodes as bn
import beanmachine.ppl.compiler.bmg_types as bt
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_node_types import is_supported_by_bmg
from beanmachine.ppl.compiler.bmg_requirements import EdgeRequirements
from beanmachine.ppl.compiler.error_report import ErrorReport, Violation
from beanmachine.ppl.compiler.graph_labels import get_edge_labels
from beanmachine.ppl.compiler.internal_error import InternalError
from beanmachine.ppl.compiler.lattice_typer import LatticeTyper


def _is_real_matrix(t: bt.BMGLatticeType) -> bool:
    return any(
        isinstance(t, m)
        for m in {
            bt.RealMatrix,
            bt.PositiveRealMatrix,
            bt.NegativeRealMatrix,
            bt.ProbabilityMatrix,
        }
    )


def _is_pos_real_matrix(t: bt.BMGLatticeType) -> bool:
    return any(
        isinstance(t, m)
        for m in {
            bt.PositiveRealMatrix,
            bt.ProbabilityMatrix,
        }
    )


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
        if r is bt.any_requirement:
            return True
        if r is bt.any_real_matrix:
            return _is_real_matrix(t)
        if r is bt.any_pos_real_matrix:
            return _is_pos_real_matrix(t)
        if r == bt.RealMatrix:
            return isinstance(t, bt.RealMatrix)
        if isinstance(r, bt.UpperBound):
            return bt.supremum(t, r.bound) == r.bound
        if isinstance(r, bt.AlwaysMatrix):
            return t == r.bound
        if r == bt.BooleanMatrix:
            return isinstance(t, bt.BooleanMatrix)
        if r == bt.ProbabilityMatrix:
            return isinstance(t, bt.ProbabilityMatrix)
        if r == bt.SimplexMatrix:
            return isinstance(t, bt.SimplexMatrix)
        return t == r

    def _node_meets_requirement(self, node: bn.BMGNode, r: bt.Requirement) -> bool:
        lattice_type = self._typer[node]
        assert lattice_type is not bt.Untypable
        if isinstance(r, bt.AlwaysMatrix):
            return self._typer.is_matrix(node) and self._type_meets_requirement(
                lattice_type, r.bound
            )
        if r is bt.any_real_matrix or r is bt.any_pos_real_matrix:
            return self._typer.is_matrix(node) and self._type_meets_requirement(
                lattice_type, r
            )
        return self._type_meets_requirement(lattice_type, r)

    def _try_to_meet_constant_requirement(
        self,
        node: bn.ConstantNode,
        requirement: bt.Requirement,
    ) -> Optional[bn.BMGNode]:
        # We have a constant node that either (1) is untyped, and therefore
        # needs to be replaced by an equivalent typed node, or (2) is typed
        # but is of the wrong type, and needs to be replaced by an equivalent
        # constant of a larger type.
        #
        # Obtain a type for the node. If the node meets an upper bound requirement
        # then it has a value that can be converted to the desired type.  If it
        # does not meet an UB requirement then there is no equivalent constant
        # node of the correct type and we give an error.
        it = self._typer[node]
        # NOTE: By this point we should have already rejected any graph that contains
        # a reachable but untypable constant node.  See comment in fix_unsupported
        # regarding UntypedConstantNode support.

        if requirement is bt.any_real_matrix:
            if _is_real_matrix(it):
                # It's already an R, R+, R- or P matrix, but it might be a single
                # value. Ensure that it is marked as a matrix, not a single value.
                assert isinstance(it, bt.BMGMatrixType)
                return self.bmg.add_constant_of_matrix_type(node.value, it)
            else:
                # It's some other type, such as Boolean or Natural matrix.
                # Emit the value as the equivalent real matrix:
                return self.bmg.add_real_matrix(node.value)

        if requirement is bt.any_pos_real_matrix:
            if _is_pos_real_matrix(it):
                assert isinstance(it, bt.BMGMatrixType)
                return self.bmg.add_constant_of_matrix_type(node.value, it)
            else:
                return self.bmg.add_pos_real_matrix(node.value)

        if requirement == bt.RealMatrix:
            if isinstance(it, bt.RealMatrix):
                return self.bmg.add_constant_of_matrix_type(node.value, it)
            else:
                return self.bmg.add_real_matrix(node.value)

        if self._type_meets_requirement(it, bt.upper_bound(requirement)):
            if requirement is bt.any_requirement:
                # The lattice type of the constant might be Zero or One; in that case,
                # generate a bool constant node.
                required_type = bt.lattice_to_bmg(it)
            else:
                required_type = bt.requirement_to_type(requirement)
            if bt.must_be_matrix(requirement):
                assert isinstance(required_type, bt.BMGMatrixType)
                result = self.bmg.add_constant_of_matrix_type(node.value, required_type)
            else:
                result = self.bmg.add_constant_of_type(node.value, required_type)
            assert self._node_meets_requirement(result, requirement)
            return result
        return None

    def _meet_constant_requirement(
        self,
        node: bn.ConstantNode,
        requirement: bt.Requirement,
        consumer: bn.BMGNode,
        edge: str,
    ) -> bn.BMGNode:

        result = self._try_to_meet_constant_requirement(node, requirement)
        if result is not None:
            return result

        # We cannot convert this node to any type that meets the requirement.
        # Add an error.
        self.errors.add_error(
            Violation(
                node,
                self._typer[node],
                requirement,
                consumer,
                edge,
                self.bmg.execution_context.node_locations(consumer),
            )
        )
        return node

    def _convert_operator_to_atomic_type(
        self,
        node: bn.OperatorNode,
        requirement: bt.BMGLatticeType,
        consumer: bn.BMGNode,
        edge: str,
    ) -> bn.BMGNode:
        # We have been given a node which does not meet a requirement,
        # but it can be converted to a node which does meet the requirement
        # that has the same semantics. Start by confirming those preconditions.
        node_type = self._typer[node]
        assert node_type != requirement
        assert bt.is_convertible_to(node_type, requirement)

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
        # node type.

        assert node_type == bt.Boolean

        # There is no "to natural" or "to probability" but since we have
        # a bool in hand, we can use an if-then-else as a conversion.

        zero = self.bmg.add_constant_of_type(0.0, requirement)
        one = self.bmg.add_constant_of_type(1.0, requirement)
        return self.bmg.add_if_then_else(node, one, zero)

    def _convert_operator_to_matrix_type(
        self,
        node: bn.OperatorNode,
        requirement: bt.Requirement,
        consumer: bn.BMGNode,
        edge: str,
    ) -> bn.BMGNode:
        if isinstance(requirement, bt.AlwaysMatrix):
            requirement = requirement.bound
        assert isinstance(requirement, bt.BMGMatrixType)

        # We have been given a node which does not meet a requirement,
        # but it can be converted to a node which does meet the requirement
        # that has the same semantics. Start by confirming those preconditions.
        node_type = self._typer[node]
        assert node_type != requirement
        assert bt.is_convertible_to(node_type, requirement)

        # Converting anything to real matrix or positive/negative real matrix is easy;
        # there's already a node for that so just insert it on the edge
        # whose requirement is not met, and the requirement will be met.

        # TODO: We do not yet handle the case where we are converting from, say,
        # an atomic probability to a 1x1 real matrix because in practice this
        # hasn't come up yet. If it does, detect it here and insert a TO_REAL
        # or TO_POS_REAL followed by a TO_MATRIX and create a test case that
        # illustrates the scenario.
        assert self._typer.is_matrix(node)

        if isinstance(requirement, bt.RealMatrix):
            return self.bmg.add_to_real_matrix(node)

        if isinstance(requirement, bt.NegativeRealMatrix):
            return self.bmg.add_to_negative_real_matrix(node)

        # TODO: We do not yet handle the case where we are converting from
        # a matrix of bools to a matrix of naturals or probabilities because
        # in practice this has not come up yet. If it does, we will need
        # to create TO_NATURAL_MATRIX and TO_PROB_MATRIX operators in BMG, or
        # come up with some other way to turn many bools into many naturals
        # or probabilities.
        assert isinstance(requirement, bt.PositiveRealMatrix)

        return self.bmg.add_to_positive_real_matrix(node)

    def _can_force_to_prob(
        self, inf_type: bt.BMGLatticeType, requirement: bt.Requirement
    ) -> bool:
        # Consider the graph created by a call like:
        #
        # Bernoulli(0.5 + some_beta() / 2)

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

    def _can_force_to_neg_real(
        self, node_type: bt.BMGLatticeType, requirement: bt.Requirement
    ) -> bool:
        # See notes in method above; we have a similar situation but for
        # negative reals.  Consider for example
        #
        # p = beta() * 0.5 + 0.4   # Sum of two probs is a positive real
        # lp = log(p)              # log of positive real is real
        # x = log1mexp(lp)         # error, log1mexp requires negative real
        #
        # Failure to deduce that p is probability leads to a seemingly
        # unrelated error later on.
        #
        # If we require a negative real but we have a real, insert a TO_NEG_REAL
        # node to do a runtime check.

        return (
            requirement == bt.NegativeReal
            or requirement == bt.upper_bound(bt.NegativeReal)
        ) and node_type == bt.Real

    def _try_to_meet_any_real_matrix_requirement(
        self,
        node: bn.OperatorNode,
        requirement: bt.Requirement,
    ) -> Optional[bn.BMGNode]:

        assert not self._node_meets_requirement(node, requirement)

        # Is the requirement that we have a real-valued matrix, but we haven't got
        # a real-valued matrix? Every value can be converted to a real-valued matrix,
        # so just insert the conversion node.

        if requirement is not bt.any_real_matrix:
            return None

        result = self.bmg.add_to_real_matrix(node)
        assert self._node_meets_requirement(result, requirement)
        return result

    def _try_to_meet_any_pos_real_matrix_requirement(
        self,
        node: bn.OperatorNode,
        requirement: bt.Requirement,
    ) -> Optional[bn.BMGNode]:

        assert not self._node_meets_requirement(node, requirement)

        # Is the requirement that we have a pos-real-valued matrix? Anything that
        # is not known to be negative can be a positive real matrix.

        if requirement is not bt.any_pos_real_matrix:
            return None

        node_type = self._typer[node]
        if isinstance(node_type, bt.NegativeRealMatrix):
            return None

        result = self.bmg.add_to_positive_real_matrix(node)
        assert self._node_meets_requirement(result, requirement)
        return result

    def _try_to_meet_upper_bound_requirement(
        self,
        node: bn.OperatorNode,
        requirement: bt.Requirement,
        consumer: bn.BMGNode,
        edge: str,
    ) -> Optional[bn.BMGNode]:

        assert not self._node_meets_requirement(node, requirement)

        node_type = self._typer[node]
        if not self._type_meets_requirement(node_type, bt.upper_bound(requirement)):
            return None

        # If we got here then the node did NOT meet the requirement,
        # but its type DID meet an upper bound requirement, which
        # implies that the requirement was not an upper bound requirement.
        assert not isinstance(requirement, bt.UpperBound)

        # We definitely can meet the requirement by inserting some sort
        # of conversion logic. We have different helper methods for
        # the atomic type and matrix type cases.
        if bt.must_be_matrix(requirement):
            result = self._convert_operator_to_matrix_type(
                node, requirement, consumer, edge
            )
        else:
            assert isinstance(requirement, bt.BMGLatticeType)
            result = self._convert_operator_to_atomic_type(
                node, requirement, consumer, edge
            )
        assert self._node_meets_requirement(result, requirement)
        return result

    def _try_to_force_to_prob(self, node, requirement) -> Optional[bn.BMGNode]:
        # We cannot make the node meet the requirement "implicitly". We can
        # "explicitly" meet a requirement of probability if we have a
        # real or pos real.

        node_type = self._typer[node]
        if not self._can_force_to_prob(node_type, requirement):
            return None
        assert node_type == bt.Real or node_type == bt.PositiveReal
        assert self._node_meets_requirement(node, node_type)
        return self.bmg.add_to_probability(node)

    def _try_to_force_to_neg_real(self, node, requirement) -> Optional[bn.BMGNode]:
        # We cannot make the node meet the requirement "implicitly". We can
        # "explicitly" meet a requirement of neg real if we have a value we do
        # not know is positive.
        node_type = self._typer[node]
        if not self._can_force_to_neg_real(node_type, requirement):
            return None

        return self.bmg.add_to_negative_real(node)

    def _try_to_meet_operator_requirement(
        self,
        node: bn.OperatorNode,
        requirement: bt.Requirement,
        consumer: bn.BMGNode,
        edge: str,
    ) -> Optional[bn.BMGNode]:
        # We should not have called this function if the input node already meets
        # the requirement on the edge.

        assert not self._node_meets_requirement(node, requirement)

        # ----
        #
        # TODO: Is the problem that we have a scalar but we need a matrix full
        # of that value?  Generate a matrix fill operation.
        #
        # TODO: Is the problem that we have a row or column matrix but we need
        # a rectangular matrix?  Generate a broadcast operation.
        #
        # TODO: Note that in either of these cases, we might *also* need to
        # generate a type conversion, so we might not meet the requirement on
        # after introducing the fill / broadcast node.
        #
        # ----

        # Is the requirement that we have a real-valued matrix, but we haven't got
        # a real-valued matrix? Every value can be converted to a real-valued matrix,
        # so that's the easiest case.  Knock it out first.

        result = self._try_to_meet_any_real_matrix_requirement(node, requirement)
        if result is not None:
            return result

        # Is the requirement that we have any positive real-valued matrix? Every value
        # except negative real scalars and matrices can be converted to a positive real
        # matrix.

        result = self._try_to_meet_any_pos_real_matrix_requirement(node, requirement)
        if result is not None:
            return result

        # If we weaken the requirement to an upper bound requirement, do we meet it? If so,
        # then there is a conversion node we can add.

        result = self._try_to_meet_upper_bound_requirement(
            node, requirement, consumer, edge
        )
        if result is not None:
            return result

        result = self._try_to_force_to_prob(node, requirement)
        if result is not None:
            return result

        result = self._try_to_force_to_neg_real(node, requirement)
        if result is not None:
            return result

        # We couldn't meet the requirement.

        return None

    def _meet_operator_requirement(
        self,
        node: bn.OperatorNode,
        requirement: bt.Requirement,
        consumer: bn.BMGNode,
        edge: str,
    ) -> bn.BMGNode:
        assert not self._node_meets_requirement(node, requirement)
        result = self._try_to_meet_operator_requirement(
            node, requirement, consumer, edge
        )
        if result is not None:
            return result

        # We were unable to meet a requirement; add an error.
        node_type = self._typer[node]
        self.errors.add_error(
            Violation(
                node,
                node_type,
                requirement,
                consumer,
                edge,
                self.bmg.execution_context.node_locations(consumer),
            )
        )
        return node

    def _check_requirement_validity(
        self,
        node: bn.BMGNode,
        requirement: bt.Requirement,
        consumer: bn.BMGNode,
        edge: str,
    ) -> None:
        ice = "Internal compiler error in edge requirements checking:\n"

        # These lattice types should never be used as requirements; a type requirement
        # must be a valid BMG type, but these are lattice types used for detecting
        # expressions which are convertible to more than one BMG type.
        if requirement in {bt.Tensor, bt.One, bt.Zero, bt.Untypable}:
            raise InternalError(
                f"{ice} Requirement {requirement} is an invalid requirement."
            )

        # We should never be checking outgoing edge requirements on a node which
        # has zero outgoing edges. If we are, something has gone wrong in the compiler.

        node_type = type(node)
        if node_type in [bn.Observation, bn.Query, bn.FactorNode]:
            raise InternalError(
                f"{ice} Node of type {node_type.__name__} is being checked for requirements but "
                + "should never have an outgoing edge '{edge}'."
            )

        # The remaining checks determine if a precondition of the requirements checker
        # is not met. It is always valid to have a constant node even if unsupported by BMG.
        # The requirements checker will replace it with an equivalent node with a valid
        # BMG type if necessary, in _meet_constant_requirement above.

        if isinstance(node, bn.ConstantNode):
            return

        # If we get here then the node is not a  constant.  Leaving aside constants, we
        # should never ask the requirements checker to consider the requirements on an
        # outgoing edge from a node that BMG does not even support.  The unsupported node
        # fixer should already have removed all such nodes.

        if not is_supported_by_bmg(node):
            raise InternalError(
                f"{ice} Node of type {node_type.__name__} is being checked for requirements but "
                + "is not supported by BMG; the unsupported node checker should already "
                + "have either replaced the node or produced an error."
            )

        # If we get here then the node is supported by BMG. The requirements checker needs to
        # know the lattice type of the node in order to check whether it meets the requirement,
        # even if the requirement is "any". If this throws then you probably need to implement
        # type analysis in the lattice typer.

        # CONSIDER: Note that we do not make a distinction here between "the lattice typer simply does
        # not have the code to type check this node" and "the lattice typer tried but failed".
        # If it is important to make that distinction then we need to have two different "untyped"
        # objects, one representing "not implemented" and one representing "failure".

        lattice_type = self._typer[node]
        if lattice_type is bt.Untypable:
            raise InternalError(
                f"{ice} Node of type {node_type.__name__} is being checked for requirements but "
                + "the lattice typer is unable to assign it a type. Requirements checking always "
                + "needs to know the lattice type of a node when checking requirements on its "
                + "outgoing edges."
            )

    def meet_requirement(
        self,
        node: bn.BMGNode,
        requirement: bt.Requirement,
        consumer: bn.BMGNode,
        edge: str,
    ) -> bn.BMGNode:
        """The consumer node consumes the value of the input node. The consumer's
        requirement is given; the name of this edge is provided for error reporting."""

        self._check_requirement_validity(node, requirement, consumer, edge)

        # If we have an untyped constant node we always need to replace it.
        if isinstance(node, bn.UntypedConstantNode):
            return self._meet_constant_requirement(node, requirement, consumer, edge)

        # If the node already meets the requirement, we're done.
        if self._node_meets_requirement(node, requirement):
            return node

        # In normal operation we should never insert a typed constant node
        # that is of the wrong type, but we have a few test cases in which
        # we do so explicitly. Regardless, it is not a problem to convert a
        # typed constant to the correct type.
        if isinstance(node, bn.ConstantNode):
            return self._meet_constant_requirement(node, requirement, consumer, edge)

        # A distribution's outgoing edges are only queries and their requirements
        # are always met, so we should have already returned. Therefore the only
        # remaining possibility is that we have an operator.
        assert isinstance(node, bn.OperatorNode)
        return self._meet_operator_requirement(node, requirement, consumer, edge)

    def fix_problems(self) -> bool:
        made_progress = False
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
                made_progress = True
        return made_progress


def requirements_fixer(bmg: BMGraphBuilder):
    rf = RequirementsFixer(bmg, LatticeTyper())
    made_progress = rf.fix_problems()
    return bmg, made_progress, rf.errors
