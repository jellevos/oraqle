"""Module containing nodes with a flexible number of operands."""
from abc import abstractmethod
from collections import Counter
from functools import reduce
from typing import Callable, override
from typing import Counter as CounterType
from typing import Dict, Optional, Set, Type

from galois import FieldArray

from oraqle.compiler.graphviz import DotFile
from oraqle.compiler.nodes.abstract import Node, UnoverloadedWrapper
from oraqle.compiler.nodes.fp.abstract import CostParetoFront
from oraqle.compiler.nodes.fp.fixed import ArithmeticNode
from oraqle.compiler.nodes.fp.leafs import FpConstant
from oraqle.compiler.nodes.fpd.abstract import FpNode


class FlexibleNode[Operand: Node](Node[Operand]):
    """A node with an arbitrary number of operands. The operation must be reducible using a binary associative operation."""

    # TODO: Ensure that when all inputs are constants, the node is replaced with its evaluation

    @override
    def arithmetize_fpd(self, strategy: str, circuit_gf: Type[FieldArray]) -> ArithmeticNode:  # noqa: D102
        if self._arithmetize_cache is None:
            self._arithmetize_cache = self._arithmetize_inner(strategy, circuit_gf)

        return self._arithmetize_cache

    @abstractmethod
    def _arithmetize_inner(self, strategy: str, circuit_gf: Type[FieldArray]) -> ArithmeticNode:
        pass

    @override
    def arithmetize_depth_aware(self, cost_of_squaring: float, circuit_gf: Type[FieldArray]) -> CostParetoFront:  # noqa: D102
        if self._arithmetize_depth_cache is None:
            self._arithmetize_depth_cache = self._arithmetize_depth_aware_inner(cost_of_squaring, circuit_gf)

        return self._arithmetize_depth_cache

    @abstractmethod
    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float, circuit_gf: Type[FieldArray]) -> CostParetoFront:
        pass


class CommutativeUniqueReducibleNode[Operand: Node](FlexibleNode[Operand]):
    """A node with an operation that is reducible without taking order into account: i.e. it has a binary operation that is associative and commutative.

    The operands are unique, i.e. the same operand will never appear twice.
    """

    def __init__(
        self,
        operands: Set[UnoverloadedWrapper[Operand]],
        #gf: Type[FieldArray],
    ):
        """Initialize a node with the given set as the operands. None of the operands can be a constant."""
        self._operands = operands
        assert not any(isinstance(operand.node, FpConstant) for operand in self._operands)
        assert len(operands) > 1
        super().__init__()

    def apply_function_to_operands(self, function: Callable[[Operand], None]):  # noqa: D102
        for operand in self._operands:
            function(operand.node)

    def replace_operands_using_function(self, function: Callable[[Operand], Operand]):  # noqa: D102
        self._operands = {UnoverloadedWrapper(function(operand.node)) for operand in self._operands}

    def evaluate(self, actual_inputs: Dict[str, FieldArray]) -> FieldArray:  # noqa: D102
        if self._evaluate_cache is None:
            self._evaluate_cache = reduce(
                self._inner_operation,
                (operand.node.evaluate(actual_inputs) for operand in self._operands),
            )

        return self._evaluate_cache

    @abstractmethod
    def _inner_operation(self, a: FieldArray, b: FieldArray) -> FieldArray:
        """Perform the reducible operation performed by this node (order should not matter)."""

    def __hash__(self) -> int:
        if self._hash is None:
            # The hash is commutative
            hashes = sorted([hash(operand) for operand in self._operands])
            self._hash = hash((self._hash_name, tuple(hashes)))

        return self._hash

    def is_equivalent(self, other: FpNode) -> bool:  # noqa: D102
        if not isinstance(other, self.__class__):
            return False

        if hash(self) != hash(other):
            return False

        return self._operands == other._operands


class CommutativeMultiplicityReducibleNode[Operand: Node](FlexibleNode[Operand]):
    """A node with an operation that is reducible without taking order into account: i.e. it has a binary operation that is associative and commutative."""

    def __init__(
        self,
        operands: CounterType[UnoverloadedWrapper],
        #gf: Type[FieldArray],
        constant: Optional[FieldArray] = None,
    ):
        """Initialize a reducible node with the given `Counter` representing the operands, none of which is allowed to be a constant."""
        super().__init__()
        self._constant = self._identity if constant is None else constant
        self._operands = operands
        assert not any(isinstance(operand, FpConstant) for operand in self._operands)
        assert (sum(operands.values()) + (self._constant != self._identity)) > 1
        assert isinstance(next(iter(self._operands)), UnoverloadedWrapper)

    @property
    @abstractmethod
    def _identity(self) -> FieldArray:
        pass

    def apply_function_to_operands(self, function: Callable[[Operand], None]):  # noqa: D102
        for operand in self._operands:
            function(operand.node)

    def replace_operands_using_function(self, function: Callable[[Operand], Operand]):  # noqa: D102
        # FIXME: What if there is only one operand remaining?
        self._operands = Counter(
            {
                UnoverloadedWrapper(function(operand.node)): count
                for operand, count in self._operands.items()
            }
        )
        assert not any(isinstance(operand.node, FpConstant) for operand in self._operands)
        assert (sum(self._operands.values()) + (self._constant != self._identity)) > 1

    def __hash__(self) -> int:
        if self._hash is None:
            # The hash is commutative
            hashes = sorted(
                [(hash(operand.node), count) for operand, count in self._operands.items()]
            )
            self._hash = hash((self._hash_name, tuple(hashes), int(self._constant)))

        return self._hash

    def is_equivalent(self, other: FpNode) -> bool:  # noqa: D102
        if not isinstance(other, self.__class__):
            return False

        if hash(self) != hash(other):
            return False

        return self._operands == other._operands and self._constant == other._constant

    def to_graph(self, graph_builder: DotFile) -> int:  # noqa: D102
        if self._to_graph_cache is None:
            super().to_graph(graph_builder)
            self._to_graph_cache: int

            if self._constant != self._identity:
                # TODO: Add known_by
                graph_builder.add_link(
                    graph_builder.add_node(label=str(self._constant)), self._to_graph_cache
                )

        return self._to_graph_cache
    

class CommutativeUniqueReducibleFpNode[Operand: Node](CommutativeUniqueReducibleNode[Operand], FpNode):

    def __init__(self, operands: Set[UnoverloadedWrapper[Operand]], gf: Type[FieldArray]):
        CommutativeUniqueReducibleNode.__init__(self, operands)
        FpNode.__init__(self, gf)


class CommutativeMultiplicityReducibleFpNode[Operand: Node](CommutativeMultiplicityReducibleNode[Operand], FpNode):

    def __init__(self, operands: Counter[UnoverloadedWrapper], gf: Type[FieldArray], constant: Optional[FieldArray] = None):
        CommutativeMultiplicityReducibleNode.__init__(self, operands, constant)
        FpNode.__init__(self, gf)
