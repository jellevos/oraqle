"""This module contains arithmetic operations between a flexible number of inputs: summations and products."""
import itertools
from collections import Counter
from dataclasses import dataclass, field
from functools import reduce
from heapq import heapify, heappop, heappush
from typing import Any
from typing import Counter as CounterType
from typing import Dict, Iterable, Optional, Tuple, Type, Union

from galois import FieldArray

from oraqle.compiler.nodes.abstract import (
    ArithmeticNode,
    CostParetoFront,
    Node,
    UnoverloadedWrapper,
    _to_node,
)
from oraqle.compiler.nodes.binary_arithmetic import Addition, Multiplication
from oraqle.compiler.nodes.flexible import CommutativeMultiplicityReducibleNode
from oraqle.compiler.nodes.leafs import Constant
from oraqle.compiler.nodes.unary_arithmetic import ConstantAddition, ConstantMultiplication


# TODO: This is mostly copied from generate_multiplication_tree (depth is different)
def _generate_addition_tree(
    summands: Iterable[Tuple[int, ArithmeticNode]], counts: Iterable[int]
) -> Tuple[int, Addition]:
    queue = [
        _PrioritizedItem(*summand) for summand, count in zip(summands, counts) for _ in range(count)
    ]
    heapify(queue)

    while len(queue) > 1:
        a = heappop(queue)
        b = heappop(queue)

        a_const = isinstance(a.item, Constant)
        b_const = isinstance(b.item, Constant)

        # TODO: This should move to Node
        if a_const:
            if b_const:
                new = a.item + b.item
            else:
                new = b.item if a.item._value == 0 else ConstantAddition(b.item, a.item._value)
        elif b_const:
            new = a.item if b.item._value == 0 else ConstantAddition(a.item, b.item._value)
        else:
            new = Addition(a.item, b.item, a.item._gf)

        heappush(
            queue,
            _PrioritizedItem(max(a.priority, b.priority), new),
        )

    return (queue[0].priority, queue[0].item)


class Sum(CommutativeMultiplicityReducibleNode):
    """This node represents a sum between two or more operands, or at least one operand and a constant."""

    @property
    def _hash_name(self) -> str:
        return "sum"

    @property
    def _node_label(self) -> str:
        return "+"

    @property
    def _identity(self) -> FieldArray:
        return self._gf(0)

    def _arithmetize_inner(self, strategy: str) -> Node:
        # TODO: Wrap exponents
        new_operands = Counter()
        new_constant = self._constant
        for operand, count in self._operands.items():
            new_operand = operand.node.arithmetize(strategy)

            if isinstance(new_operand, Constant):
                new_constant += new_operand._value * count
            else:
                new_operands[UnoverloadedWrapper(new_operand)] += count

        if len(new_operands) == 0:
            return Constant(new_constant)  # type: ignore
        elif sum(new_operands.values()) == 1 and new_constant == self._identity:
            return next(iter(new_operands)).node

        return Sum(new_operands, self._gf, new_constant)

    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        # FIXME: This could be done way more efficiently by iterating over increasing depth
        front = CostParetoFront(cost_of_squaring)

        for operands in itertools.product(
            *(
                operand.node.arithmetize_depth_aware(cost_of_squaring)
                for operand in self._operands
            )
        ):
            addition_tree = _generate_addition_tree(
                ((d, operand) for d, _, operand in operands), self._operands.values()
            )
            if self._constant != self._identity:
                if isinstance(addition_tree[1], Constant):
                    return CostParetoFront.from_leaf(
                        Constant(addition_tree[1]._value + self._constant), cost_of_squaring
                    )

                addition_tree = (
                    addition_tree[0],
                    ConstantAddition(addition_tree[1], self._constant),
                )
            front.add(addition_tree[1], depth=addition_tree[0])

        assert not front.is_empty()
        return front

    def to_arithmetic(self) -> ArithmeticNode:  # noqa: D102
        if self._arithmetic_cache is None:
            # FIXME: Perform actual rebalancing
            operands = iter(self._operands.elements())

            # TODO: There is a lot of duplication between this and multiplications
            if self._constant == self._identity:
                self._arithmetic_cache = Addition(
                    next(operands).node.to_arithmetic(),
                    next(operands).node.to_arithmetic(),
                    self._gf,
                )
            else:
                self._arithmetic_cache = ConstantAddition(
                    next(operands).node.to_arithmetic(), self._constant
                )

            for operand in operands:
                self._arithmetic_cache = Addition(
                    self._arithmetic_cache, operand.node.to_arithmetic(), self._gf
                )

        return self._arithmetic_cache

    def evaluate(self, actual_inputs: Dict[str, FieldArray]) -> FieldArray:  # noqa: D102
        if self._evaluate_cache is None:
            self._evaluate_cache = reduce(
                lambda a, b: a + b,
                (
                    operand.node.evaluate(actual_inputs) * count
                    for operand, count in self._operands.items()
                ),
            )
            self._evaluate_cache += self._constant

        return self._evaluate_cache  # type: ignore

    def add_flatten(self, other: Node) -> Node:
        """Adds this node to `other`, flattening the summation if either of the two is also a `Sum` and absorbing `Constant`s.
        
        Returns:
            A `Sum` node containing the flattened summation, or a `Constant` node.
        """
        order = self._gf.order
        # TODO: Consider already assigning values to e.g. result._depth
        if isinstance(other, Sum):
            counter = self._operands + other._operands
            counter_dict = {
                el: count % order for el, count in counter.items() if count % order != 0
            }
            constant = self._constant + other._constant
            if len(counter_dict) == 0:
                return Constant(constant)  # type: ignore
            return Sum(Counter(counter_dict), self._gf, constant)  # type: ignore
        elif isinstance(other, Constant):
            if sum(self._operands.values()) == 1 and int(self._constant + other._value) == 0:
                return next(iter(self._operands)).node
            return Sum(self._operands, self._gf, self._constant + other._value)  # type: ignore

        counter = self._operands.copy()
        unoverloaded_other = UnoverloadedWrapper(other)
        counter[unoverloaded_other] = (counter[unoverloaded_other] + 1) % order
        if counter[unoverloaded_other] == 0:
            counter.pop(unoverloaded_other)

        # FIXME: If empty, return Constant(0)

        return Sum(counter, self._gf, self._constant)


@dataclass(order=True)
class _PrioritizedItem:
    priority: int
    item: Any = field(compare=False)


def _generate_multiplication_tree(
    multiplicands: Iterable[Tuple[int, ArithmeticNode]], counts: Iterable[int]
) -> Tuple[int, Multiplication]:
    queue = [
        _PrioritizedItem(*multiplicand)
        for multiplicand, count in zip(multiplicands, counts)
        for _ in range(count)
    ]
    heapify(queue)

    while len(queue) > 1:
        a = heappop(queue)
        b = heappop(queue)

        a_const = isinstance(a.item, Constant)
        b_const = isinstance(b.item, Constant)

        # TODO: This should move to Node
        if a_const:
            if b_const:
                new = a.item * b.item
            elif a.item._value == 1:
                new = b.item
            else:
                new = ConstantMultiplication(b.item, a.item._value)
        elif b_const:
            new = a.item if b.item._value == 1 else ConstantMultiplication(a.item, b.item._value)
        else:
            new = Multiplication(a.item, b.item, a.item._gf)

        heappush(
            queue,
            _PrioritizedItem(max(a.priority, b.priority) + (not a_const and not b_const), new),
        )

    return (queue[0].priority, queue[0].item)


class Product(CommutativeMultiplicityReducibleNode):
    """This node represents a product between two or more operands, or at least one operand and a constant."""

    def __init__(
        self,
        operands: CounterType[UnoverloadedWrapper],
        gf: Type[FieldArray],
        constant: Optional[FieldArray] = None,
    ):
        """Initialize a `Product` with the given `Counter` as operands and an optional `constant`."""
        super().__init__(operands, gf, constant)
        assert constant != 0

    @property
    def _hash_name(self) -> str:
        return "product"

    @property
    def _node_label(self) -> str:
        return "×"  # noqa: RUF001

    @property
    def _identity(self) -> FieldArray:
        return self._gf(1)

    def _inner_operation(self, a: FieldArray, b: FieldArray) -> FieldArray:
        return a * b  # type: ignore

    def _arithmetize_inner(self, strategy: str) -> Node:
        # TODO: Wrap exponents
        new_operands = Counter()
        new_constant = self._constant
        for operand, count in self._operands.items():
            new_operand = operand.node.arithmetize(strategy)

            if isinstance(new_operand, Constant):
                new_constant *= new_operand._value**count
            else:
                new_operands[UnoverloadedWrapper(new_operand)] += count

        if len(new_operands) == 0:
            return Constant(new_constant)  # type: ignore
        elif sum(new_operands.values()) == 1 and new_constant == self._identity:
            return next(iter(new_operands)).node

        if new_constant == 0:
            return Constant(self._gf(0))

        return Product(new_operands, self._gf, new_constant)  # type: ignore

    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        # TODO: This could be done more efficiently by going breadth-wise
        front = CostParetoFront(cost_of_squaring)

        for operands in itertools.product(
            *(
                operand.node.arithmetize_depth_aware(cost_of_squaring)
                for operand in self._operands
            )
        ):
            multiplication_tree = _generate_multiplication_tree(
                ((d, operand) for d, _, operand in operands), self._operands.values()
            )
            if self._constant != self._identity:
                if isinstance(multiplication_tree[1], Constant):
                    return CostParetoFront.from_leaf(
                        Constant(multiplication_tree[1]._value * self._constant), cost_of_squaring
                    )

                multiplication_tree = (
                    multiplication_tree[0],
                    ConstantMultiplication(multiplication_tree[1], self._constant),
                )
            front.add(multiplication_tree[1], depth=multiplication_tree[0])

        assert not front.is_empty()
        return front

    def to_arithmetic(self) -> ArithmeticNode:  # noqa: D102
        if self._arithmetic_cache is None:
            # FIXME: Perform actual rebalancing
            operands = iter(self._operands.elements())

            if self._constant == self._identity:
                self._arithmetic_cache = Multiplication(
                    next(operands).node.to_arithmetic(),
                    next(operands).node.to_arithmetic(),
                    self._gf,
                )
            else:
                self._arithmetic_cache = ConstantMultiplication(
                    next(operands).node.to_arithmetic(), self._constant
                )

            for operand in operands:
                self._arithmetic_cache = Multiplication(
                    self._arithmetic_cache, operand.node.to_arithmetic(), self._gf
                )

        return self._arithmetic_cache

    def evaluate(self, actual_inputs: Dict[str, FieldArray]) -> FieldArray:  # noqa: D102
        if self._evaluate_cache is None:
            self._evaluate_cache = reduce(lambda a, b: a * b, (operand.node.evaluate(actual_inputs) ** count for operand, count in self._operands.items()))  # type: ignore
            self._evaluate_cache *= self._constant  # type: ignore

        return self._evaluate_cache  # type: ignore

    def mul_flatten(self, other: Node) -> Node:
        """Multiplies this node with `other`, flattening the product if either of the two is also a `Product` and absorbing `Constant`s.
        
        Returns:
            A `Product` node containing the flattened product, or a `Constant` node.
        """
        # TODO: Consider already assigning values to e.g. result._depth
        if isinstance(other, Product):
            # TODO: Wrap powers (due to modulo arithmetic)
            return Product(self._operands + other._operands, self._gf, self._constant * other._constant)  # type: ignore
        elif isinstance(other, Constant):
            if other._value == 0:
                return Constant(self._gf(0))
            return Product(self._operands, self._gf, self._constant * other._value)  # type: ignore

        counter = self._operands.copy()
        counter[UnoverloadedWrapper(other)] += 1  # type: ignore
        return Product(counter, self._gf, self._constant)


def _first_gf(*operands: Union[Node, int, bool]) -> Optional[Type[FieldArray]]:
    for operand in operands:
        if isinstance(operand, Node):
            return operand._gf


def sum_(*operands: Union[Node, int, bool]) -> Sum:
    """Performs a sum between any number of nodes (or operands such as integers).
    
    Returns:
        A `Sum` between all operands.
    """
    assert len(operands) > 0
    gf = _first_gf(*operands)
    assert gf is not None
    return Sum(Counter(UnoverloadedWrapper(_to_node(operand, gf)) for operand in operands), gf)


def product_(*operands: Node) -> Product:
    """Performs a product between any number of nodes (or operands such as integers).
    
    Returns:
        A `Product` between all operands.
    """
    assert len(operands) > 0
    gf = _first_gf(*operands)
    assert gf is not None
    return Product(Counter(UnoverloadedWrapper(_to_node(operand, gf)) for operand in operands), gf)
