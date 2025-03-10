from abc import abstractmethod
from typing import Callable, Dict, List, override

from galois import FieldArray
from oraqle.compiler.nodes.abstract import Node


class FixedNode[Operand: Node](Node[Operand]):
    """A node with a fixed number of operands."""

    @abstractmethod
    def operands(self) -> List[Operand]:
        """Returns the operands (children) of this node. The list can be empty."""

    @abstractmethod
    def set_operands(self, operands: List[Operand]):
        """Overwrites the operands of this node."""
        # TODO: Consider replacing this method with a graph traversal method that applies a function on all operands and replaces them.

    def evaluate(self, actual_inputs: Dict[str, FieldArray]) -> FieldArray:  # noqa: D102
        # TODO: Remove modulus in this method and store it in each node instead. Alternatively, add `modulus` to methods such as `flatten` as well.
        if self._evaluate_cache is None:
            self._evaluate_cache = self.operation(
                [operand.evaluate(actual_inputs) for operand in self.operands()]
            )

        return self._evaluate_cache

    @abstractmethod
    def operation(self, operands: List[FieldArray]) -> FieldArray:
        """Evaluates this node on the specified operands."""

    def apply_function_to_operands(self, function: Callable[[Operand], None]):  # noqa: D102
        for operand in self.operands():
            function(operand)
    
    def replace_operands_using_function(self, function: Callable[[Operand], Operand]):  # noqa: D102
        self.set_operands([function(operand) for operand in self.operands()])
        # TODO: These caches should only be cleared if this is an ArithmeticNode
        self._multiplications = None
        self._squarings = None
        self._depth_cache = None


class LeafNode(FixedNode):

    def operands(self) -> List[Node]:  # noqa: D102
        return []

    def set_operands(self, operands: List[Node]):  # noqa: D102
        pass


class BinaryNode[Operand: Node](FixedNode[Operand]):

    def __init__(self,
        left: Operand,
        right: Operand) -> None:
        super().__init__()

    def operands(self) -> List[Operand]:  # noqa: D102
        return [self._left, self._right]

    def set_operands(self, operands: List[Operand]):  # noqa: D102
        self._left = operands[0]
        self._right = operands[1]

    @override
    def operation(self, operands: List[FieldArray]) -> FieldArray:
        return self._operation_inner(operands[0], operands[1])

    @abstractmethod    
    def _operation_inner(self, left: FieldArray, right: FieldArray) -> FieldArray:
        pass

    def __hash__(self) -> int:
        if self._hash is None:
            left_hash = hash(self._left)
            right_hash = hash(self._right)

            # Make the hash commutative
            if left_hash < right_hash:
                self._hash = hash((self._hash_name, (left_hash, right_hash)))
            else:
                self._hash = hash((self._hash_name, (right_hash, left_hash)))

        return self._hash

    def is_equivalent(self, other: Operand) -> bool:  # noqa: D102
        if not isinstance(other, self.__class__):
            return False

        if hash(self) != hash(other):
            return False

        # Equivalence by commutative equality
        return (
            self._left.is_equivalent(other._left) and self._right.is_equivalent(other._right)
        ) or (self._left.is_equivalent(other._right) and self._right.is_equivalent(other._left))
