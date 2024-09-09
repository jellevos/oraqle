"""Module containing binary arithmetic nodes: additions and multiplications between non-constant nodes."""
from abc import abstractmethod
from typing import List, Optional, Set, Tuple, Type

from galois import FieldArray

from oraqle.compiler.instructions import (
    AdditionInstruction,
    ArithmeticInstruction,
    MultiplicationInstruction,
)
from oraqle.compiler.nodes.abstract import (
    ArithmeticNode,
    CostParetoFront,
    Node,
    iterate_increasing_depth,
    select_stack_index,
)
from oraqle.compiler.nodes.fixed import BinaryNode
from oraqle.compiler.nodes.leafs import Constant


class CommutativeBinaryNode(BinaryNode):
    """This node has two operands and implements a commutative operation between arithmetic nodes."""

    def __init__(
        self,
        left: Node,
        right: Node,
        gf: Type[FieldArray],
    ):
        """Initialize the binary node with operands `left` and `right`."""
        self._left = left
        self._right = right
        super().__init__(gf)

    @abstractmethod
    def _operation_inner(self, x: FieldArray, y: FieldArray) -> FieldArray:
        """Applies the binary operation on x and y."""

    def operation(self, operands: List[FieldArray]) -> FieldArray:  # noqa: D102
        return self._operation_inner(operands[0], operands[1])

    def operands(self) -> List[Node]:  # noqa: D102
        return [self._left, self._right]

    def set_operands(self, operands: List[ArithmeticNode]):  # noqa: D102
        self._left = operands[0]
        self._right = operands[1]

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

    def is_equivalent(self, other: Node) -> bool:  # noqa: D102
        if not isinstance(other, self.__class__):
            return False

        if hash(self) != hash(other):
            return False

        # Equivalence by commutative equality
        return (
            self._left.is_equivalent(other._left) and self._right.is_equivalent(other._right)
        ) or (self._left.is_equivalent(other._right) and self._right.is_equivalent(other._left))


class CommutativeArithmeticBinaryNode(CommutativeBinaryNode):
    """This node has two operands and implements a commutative operation between arithmetic nodes."""

    def __init__(
        self,
        left: ArithmeticNode,
        right: ArithmeticNode,
        gf: Type[FieldArray],
    ):
        """Initialize this binary node with the given `left` and `right` operands.
        
        Raises:
            Exception: Neither `left` nor `right` is allowed to be a `Constant`.
        """
        super().__init__(left, right, gf)

        self._multiplications: Optional[Set[int]] = None
        self._squarings: Optional[Set[int]] = None
        self._depth_cache: Optional[int] = None

        if isinstance(left, Constant) or isinstance(right, Constant):
            self._is_multiplication = False
            raise Exception("This should be a constant.")

    def multiplicative_depth(self) -> int:  # noqa: D102
        if self._depth_cache is None:
            self._depth_cache = self._is_multiplication + max(
                self._left.multiplicative_depth(), self._right.multiplicative_depth()
            )

        return self._depth_cache

    def multiplications(self) -> Set[int]:  # noqa: D102
        if self._multiplications is None:
            self._multiplications = set().union(
                *(operand.multiplications() for operand in self.operands())  # type: ignore
            )
            if self._is_multiplication:
                self._multiplications.add(id(self))

        return self._multiplications

    # TODO: Squaring should probably be a UniveriateNode
    def squarings(self) -> Set[int]:  # noqa: D102
        if self._squarings is None:
            self._squarings = set().union(*(operand.squarings() for operand in self.operands()))  # type: ignore
            if self._is_multiplication and id(self._left) == id(self._right):
                self._squarings.add(id(self))

        return self._squarings

    def create_instructions(  # noqa: D102
        self,
        instructions: List[ArithmeticInstruction],
        stack_counter: int,
        stack_occupied: List[bool],
    ) -> Tuple[int, int]:
        self._left: ArithmeticNode
        self._right: ArithmeticNode

        if self._instruction_cache is None:
            left_index, stack_counter = self._left.create_instructions(
                instructions, stack_counter, stack_occupied
            )
            right_index, stack_counter = self._right.create_instructions(
                instructions, stack_counter, stack_occupied
            )

            # FIXME: Is it possible for e.g. self._left._instruction_cache to be None?

            self._left._parent_count -= 1
            if self._left._parent_count == 0:
                stack_occupied[self._left._instruction_cache] = False  # type: ignore

            self._right._parent_count -= 1
            if self._right._parent_count == 0:
                stack_occupied[self._right._instruction_cache] = False  # type: ignore

            self._instruction_cache = select_stack_index(stack_occupied)

            if self._is_multiplication:
                instructions.append(
                    MultiplicationInstruction(self._instruction_cache, left_index, right_index)
                )
            else:
                instructions.append(
                    AdditionInstruction(self._instruction_cache, left_index, right_index)
                )

        return self._instruction_cache, stack_counter


# FIXME: This order should probably change
class Addition(CommutativeArithmeticBinaryNode, ArithmeticNode):
    """Performs modular addition of two previous nodes in an arithmetic circuit."""

    @property
    def _overriden_graphviz_attributes(self) -> dict:
        return {"shape": "square", "style": "rounded,filled", "fillcolor": "grey80"}

    @property
    def _hash_name(self) -> str:
        return "add"

    @property
    def _node_label(self) -> str:
        return "+"

    def __init__(
        self,
        left: ArithmeticNode,
        right: ArithmeticNode,
        gf: Type[FieldArray],
    ):
        """Initialize a modular addition between `left` and `right`."""
        self._is_multiplication = False
        super().__init__(left, right, gf)

    def _operation_inner(self, x, y):
        return x + y

    def arithmetize(self, strategy: str) -> Node:  # noqa: D102
        self._left = self._left.arithmetize(strategy)
        self._right = self._right.arithmetize(strategy)
        return self

    def _arithmetize_inner(self, strategy: str) -> Node:
        raise NotImplementedError()

    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        front = CostParetoFront(cost_of_squaring)

        for res1, res2 in iterate_increasing_depth(
            self._left.arithmetize_depth_aware(cost_of_squaring),
            self._right.arithmetize_depth_aware(cost_of_squaring),
        ):
            d1, _, e1 = res1
            d2, _, e2 = res2

            # TODO: Do we use + here for flattening?
            front.add(Addition(e1, e2, self._gf), depth=max(d1, d2))

        assert not front.is_empty()
        return front


class Multiplication(CommutativeArithmeticBinaryNode, ArithmeticNode):
    """Performs modular multiplication of two previous nodes in an arithmetic circuit."""

    @property
    def _overriden_graphviz_attributes(self) -> dict:
        return {"shape": "square", "style": "rounded,filled", "fillcolor": "lightpink"}

    @property
    def _hash_name(self) -> str:
        return "mul"

    @property
    def _node_label(self) -> str:
        return "×"  # noqa: RUF001

    def __init__(
        self,
        left: ArithmeticNode,
        right: ArithmeticNode,
        gf: Type[FieldArray],
    ):
        """Initialize a modular multiplication between `left` and `right`."""
        assert isinstance(left, ArithmeticNode)
        assert isinstance(right, ArithmeticNode)

        self._is_multiplication = True
        super().__init__(left, right, gf)

    def _operation_inner(self, x, y):
        return x * y

    # TODO: This is very hacky! Arithmetic nodes should simply not have to be arithmetized...
    def arithmetize(self, strategy: str) -> Node:  # noqa: D102
        self._left = self._left.arithmetize(strategy)
        self._right = self._right.arithmetize(strategy)
        return self

    def _arithmetize_inner(self, strategy: str) -> Node:
        raise NotImplementedError()

    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        return CostParetoFront.from_node(self, cost_of_squaring)
