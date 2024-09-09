"""This module contains tools for evaluating AND operations between many inputs."""
import itertools
import math
from abc import ABC, abstractmethod
from collections import Counter
from heapq import heapify, heappop, heappush
from typing import Iterable, List, Optional, Sequence, Set, Tuple, Type

from galois import GF, FieldArray

from oraqle.add_chains.addition_chains_front import gen_pareto_front
from oraqle.add_chains.addition_chains_mod import chain_cost
from oraqle.add_chains.solving import extract_indices
from oraqle.compiler.boolean.bool_neg import Neg
from oraqle.compiler.comparison.equality import IsNonZero
from oraqle.compiler.nodes.abstract import (
    ArithmeticNode,
    CostParetoFront,
    Node,
    UnoverloadedWrapper,
)
from oraqle.compiler.nodes.arbitrary_arithmetic import (
    _PrioritizedItem,
    Product,
    Sum,
    _generate_multiplication_tree,
)
from oraqle.compiler.nodes.binary_arithmetic import Multiplication
from oraqle.compiler.nodes.flexible import CommutativeUniqueReducibleNode
from oraqle.compiler.nodes.leafs import Constant, Input


class And(CommutativeUniqueReducibleNode):
    """Performs an AND operation over several operands. The user must ensure that the operands are Booleans."""

    @property
    def _hash_name(self) -> str:
        return "and"

    @property
    def _node_label(self) -> str:
        return "AND"

    def _inner_operation(self, a: FieldArray, b: FieldArray) -> FieldArray:
        return self._gf(bool(a) & bool(b))

    def _arithmetize_inner(self, strategy: str) -> Node:  # noqa: PLR0911, PLR0912
        new_operands: Set[UnoverloadedWrapper] = set()
        for operand in self._operands:
            new_operand = operand.node.arithmetize(strategy)

            if isinstance(new_operand, Constant):
                if not bool(new_operand._value):
                    return Constant(self._gf(0))
                continue

            new_operands.add(UnoverloadedWrapper(new_operand))

        if len(new_operands) == 0:
            return Constant(self._gf(1))
        elif len(new_operands) == 1:
            return next(iter(new_operands)).node

        if strategy == "naive":
            return Product(Counter({operand: 1 for operand in new_operands}), self._gf).arithmetize(
                strategy
            )

        # TODO: Calling to_arithmetic here should not be necessary if we can decide the predicted depth
        queue = [
            (
                _PrioritizedItem(
                    0, operand.node
                )  # TODO: We should just maybe make a breadth method on Node
                if isinstance(operand.node, Constant)
                else _PrioritizedItem(
                    operand.node.to_arithmetic().multiplicative_depth(), operand.node
                )
            )
            for operand in new_operands
        ]
        heapify(queue)

        while len(queue) > (self._gf._characteristic - 1):
            total_sum = None
            max_depth = None
            for _ in range(self._gf._characteristic - 1):
                if len(queue) == 0:
                    break

                popped = heappop(queue)
                if max_depth is None or max_depth < popped.priority:
                    max_depth = popped.priority

                if total_sum is None:
                    total_sum = Neg(popped.item, self._gf)
                else:
                    total_sum += Neg(popped.item, self._gf)

            assert total_sum is not None
            final_result = Neg(IsNonZero(total_sum, self._gf), self._gf).arithmetize(strategy)

            assert max_depth is not None
            heappush(queue, _PrioritizedItem(max_depth, final_result))

        if len(queue) == 1:
            return heappop(queue).item

        dummy_node = Input("dummy_node", self._gf)
        is_non_zero = IsNonZero(dummy_node, self._gf).arithmetize(strategy).to_arithmetic()
        cost = is_non_zero.multiplicative_cost(
            1.0
        )  # FIXME: This needs to be the actual squaring cost

        if len(queue) - 1 < cost:
            return Product(
                Counter({UnoverloadedWrapper(operand.item): 1 for operand in queue}), self._gf
            ).arithmetize(strategy)

        return Neg(
            IsNonZero(
                Sum(
                    Counter({UnoverloadedWrapper(Neg(node.item, self._gf)): 1 for node in queue}),
                    self._gf,
                ),
                self._gf,
            ),
            self._gf,
        ).arithmetize(strategy)

    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        new_operands: Set[CostParetoFront] = set()
        for operand in self._operands:
            new_operand = operand.node.arithmetize_depth_aware(cost_of_squaring)
            new_operands.add(new_operand)

        if len(new_operands) == 0:
            return CostParetoFront.from_leaf(Constant(self._gf(1)), cost_of_squaring)
        elif len(new_operands) == 1:
            return next(iter(new_operands))

        front = CostParetoFront(cost_of_squaring)

        # TODO: This is brute force composition
        for operands in itertools.product(*(iter(new_operand) for new_operand in new_operands)):
            checked_operands = []
            for depth, cost, node in operands:
                if isinstance(node, Constant):
                    assert int(node._value) in {0, 1}
                    if node._value == 0:
                        return CostParetoFront.from_leaf(Constant(self._gf(0)), cost_of_squaring)
                else:
                    checked_operands.append((depth, cost, node))

            if len(checked_operands) == 0:
                return CostParetoFront.from_leaf(Constant(self._gf(1)), cost_of_squaring)

            if len(checked_operands) == 1:
                depth, cost, node = checked_operands[0]
                front.add(node, depth, cost)
                continue

            this_front = _find_depth_cost_front(
                checked_operands,
                self._gf,
                float("inf"),
                squaring_cost=cost_of_squaring,
                is_and=True,
            )
            front.add_front(this_front)

        return front

    def and_flatten(self, other: Node) -> Node:
        """Performs an AND operation with `other`, flattening the `And` node if either of the two is also an `And` and absorbing `Constant`s.
        
        Returns:
            An `And` node containing the flattened AND operation, or a `Constant` node.
        """
        if isinstance(other, Constant):
            if bool(other._value):
                return self
            else:
                return Constant(self._gf(0))

        if isinstance(other, And):
            return And(self._operands | other._operands, self._gf)

        new_operands = self._operands.copy()
        new_operands.add(UnoverloadedWrapper(other))
        return And(new_operands, self._gf)


def test_evaluate_mod3():  # noqa: D103
    gf = GF(3)

    a = Input("a", gf)
    b = Input("b", gf)
    node = (a & b).arithmetize("best-effort")

    assert node.evaluate({"a": gf(0), "b": gf(0)}) == gf(0)
    node.clear_cache(set())
    assert node.evaluate({"a": gf(0), "b": gf(1)}) == gf(0)
    node.clear_cache(set())
    assert node.evaluate({"a": gf(1), "b": gf(0)}) == gf(0)
    node.clear_cache(set())
    assert node.evaluate({"a": gf(1), "b": gf(1)}) == gf(1)


def test_evaluate_arithmetized_mod3():  # noqa: D103
    gf = GF(3)

    a = Input("a", gf)
    b = Input("b", gf)
    node = (a & b).arithmetize("best-effort")

    node.clear_cache(set())
    assert node.evaluate({"a": gf(0), "b": gf(0)}) == gf(0)
    node.clear_cache(set())
    assert node.evaluate({"a": gf(0), "b": gf(1)}) == gf(0)
    node.clear_cache(set())
    assert node.evaluate({"a": gf(1), "b": gf(0)}) == gf(0)
    node.clear_cache(set())
    assert node.evaluate({"a": gf(1), "b": gf(1)}) == gf(1)


def test_evaluate_arithmetized_depth_aware_mod2():  # noqa: D103
    gf = GF(2)

    a = Input("a", gf)
    b = Input("b", gf)
    node = a & b
    front = node.arithmetize_depth_aware(cost_of_squaring=1.0)

    for _, _, n in front:
        n.clear_cache(set())
        assert n.evaluate({"a": gf(0), "b": gf(0)}) == gf(0)
        n.clear_cache(set())
        assert n.evaluate({"a": gf(0), "b": gf(1)}) == gf(0)
        n.clear_cache(set())
        assert n.evaluate({"a": gf(1), "b": gf(0)}) == gf(0)
        n.clear_cache(set())
        assert n.evaluate({"a": gf(1), "b": gf(1)}) == gf(1)


def test_evaluate_arithmetized_depth_aware_mod3():  # noqa: D103
    gf = GF(3)

    a = Input("a", gf)
    b = Input("b", gf)
    node = a & b
    front = node.arithmetize_depth_aware(cost_of_squaring=1.0)

    for _, _, n in front:
        n.clear_cache(set())
        assert n.evaluate({"a": gf(0), "b": gf(0)}) == gf(0)
        n.clear_cache(set())
        assert n.evaluate({"a": gf(0), "b": gf(1)}) == gf(0)
        n.clear_cache(set())
        assert n.evaluate({"a": gf(1), "b": gf(0)}) == gf(0)
        n.clear_cache(set())
        assert n.evaluate({"a": gf(1), "b": gf(1)}) == gf(1)


def test_evaluate_arithmetized_depth_aware_7_mod5():  # noqa: D103
    gf = GF(5)

    xs = {Input(f"x{i}", gf) for i in range(7)}
    node = And({UnoverloadedWrapper(x) for x in xs}, gf)  # type: ignore
    front = node.arithmetize_depth_aware(cost_of_squaring=1.0)

    for _, _, n in front:
        n.clear_cache(set())
        assert n.evaluate({f"x{i}": gf(0) for i in range(50)}) == gf(0)
        n.clear_cache(set())
        assert n.evaluate({f"x{i}": gf(i % 2) for i in range(50)}) == gf(0)
        n.clear_cache(set())
        assert n.evaluate({f"x{i}": gf(1) for i in range(50)}) == gf(1)


def test_evaluate_arithmetized_depth_aware_50_mod31():  # noqa: D103
    gf = GF(31)

    xs = {Input(f"x{i}", gf) for i in range(50)}
    node = And({UnoverloadedWrapper(x) for x in xs}, gf)  # type: ignore
    front = node.arithmetize_depth_aware(cost_of_squaring=1.0)

    for _, _, n in front:
        n.clear_cache(set())
        assert n.evaluate({f"x{i}": gf(0) for i in range(50)}) == gf(0)
        n.clear_cache(set())
        assert n.evaluate({f"x{i}": gf(i % 2) for i in range(50)}) == gf(0)
        n.clear_cache(set())
        assert n.evaluate({f"x{i}": gf(1) for i in range(50)}) == gf(1)


class NaryLogicNode(ABC):
    """Represents a (sub)circuit for computing and AND or OR operation."""

    def __init__(self, breadth: int, cost: float) -> None:
        """Initialize this logic node with the given `breadth` and `cost` (which are not checked)."""
        self.breadth = breadth
        self.cost = cost

    @abstractmethod
    def local_cost(self) -> float:
        """Compute the local multiplicative cost, so ignoring the cost of the inputs."""

    @abstractmethod
    def print(self, level: int = 0):
        """Prints this subcircuit for debugging purposes."""

    @abstractmethod
    def to_arithmetic_node(self, is_and: bool, gf: Type[FieldArray]) -> ArithmeticNode:
        """Returns an `ArithmeticNode` representing this logic node (AND if `is_and = True` else OR)."""


class InputNaryLogicNode(NaryLogicNode):
    """An input logic node."""

    def __init__(self, node: ArithmeticNode, breadth: int) -> None:
        """Initialize the input node with the given `breadth`."""
        self._node = node
        super().__init__(breadth, 0.0)

    def local_cost(self) -> float:  # noqa: D102
        return 0.0

    def print(self, level: int = 0):  # noqa: D102
        print("  " * level + "x")

    def to_arithmetic_node(self, is_and: bool, gf: Type[FieldArray]) -> ArithmeticNode:  # noqa: D102
        return self._node


class ProductNaryLogicNode(NaryLogicNode):
    """A `ProductNaryLogicNode` represents an OR/AND (sub)circuit in which all inputs are multiplied (and flattened)."""

    def __init__(self, operands: List[NaryLogicNode], breadth: int) -> None:
        """Initialize a product subcircuit with the given `operands` and `breadth`."""
        # Merge subproducts into this product
        self._operands = list(
            itertools.chain.from_iterable(
                operand._operands if isinstance(operand, ProductNaryLogicNode) else [operand]
                for operand in operands
            )
        )
        self._arithmetic_node = None
        self._is_and = None
        super().__init__(breadth, self._compute_cost())

    def _compute_cost(self) -> float:
        return sum(op.cost for op in self._operands) + len(self._operands) - 1

    def local_cost(self) -> float:  # noqa: D102
        return len(self._operands) - 1

    def print(self, level: int = 0):  # noqa: D102
        print("  " * level + "prod:")
        for op in self._operands:
            op.print(level + 1)

    def to_arithmetic_node(self, is_and: bool, gf: Type[FieldArray]) -> ArithmeticNode:  # noqa: D102
        if self._is_and is not None and self._is_and != is_and:
            self._arithmetic_node = None

        if self._arithmetic_node is None:
            _, result = _generate_multiplication_tree(((math.ceil(math.log2(operand.breadth)), operand.to_arithmetic_node(is_and, gf) if is_and else Neg(operand.to_arithmetic_node(is_and, gf), gf).arithmetize("best-effort").to_arithmetic()) for operand in self._operands), (1 for _ in range(len(self._operands))))  # type: ignore

            if not is_and:
                result = Neg(result, gf)

            self._arithmetic_node = result.arithmetize(
                "best-effort"
            ).to_arithmetic()  # TODO: This could be more elegant
            self._is_and = is_and

        assert math.ceil(math.log2(self.breadth)) == self._arithmetic_node.multiplicative_depth()  # type: ignore
        return self._arithmetic_node


class SumReduceNaryLogicNode(NaryLogicNode):
    """A `SumReduceNaryLogicNode` represents an OR/AND (sub)circuit in which all inputs are summed and then reduced to a Boolean."""

    def __init__(
        self,
        operands: List[NaryLogicNode],
        exponentiation_depth: int,
        exponentiation_cost: float,
        exponentiation_chain: List[Tuple[int, int]],
        breadth: int,
    ) -> None:
        """Initialize a sum-reduce subcircuit with the given exponentiation chain (and properties), over the given `operands`."""
        self._operands = operands
        self._exponentiation_depth = exponentiation_depth
        self._exponentiation_cost = exponentiation_cost
        self._exponentiation_chain = exponentiation_chain
        self._arithmetic_node = None
        self._is_and = None
        super().__init__(breadth, self._compute_cost())

    def _compute_cost(self) -> float:
        return sum(op.cost for op in self._operands) + self._exponentiation_cost

    def local_cost(self) -> float:  # noqa: D102
        return self._exponentiation_cost

    def print(self, level: int = 0):  # noqa: D102
        print("  " * level + f"sumred({self._exponentiation_depth}, {self._exponentiation_cost}):")
        for op in self._operands:
            op.print(level + 1)

    def to_arithmetic_node(self, is_and: bool, gf: Type[FieldArray]) -> ArithmeticNode:  # noqa: D102
        if self._is_and is not None and self._is_and != is_and:
            self._arithmetic_node = None

        if self._arithmetic_node is None:
            # TODO: This should be replaced by augmented circuit nodes
            if is_and:
                result = (
                    Sum(
                        Counter(
                            {
                                UnoverloadedWrapper(
                                    Neg(operand.to_arithmetic_node(is_and, gf), gf)
                                ): 1
                                for operand in self._operands
                            }
                        ),
                        gf,
                    )
                    .arithmetize("best-effort")
                    .to_arithmetic()
                )
            else:
                result = (
                    Sum(
                        Counter(
                            {
                                UnoverloadedWrapper(operand.to_arithmetic_node(is_and, gf)): 1
                                for operand in self._operands
                            }
                        ),
                        gf,
                    )
                    .arithmetize("best-effort")
                    .to_arithmetic()
                )

            # Exponentiation
            chain = extract_indices(self._exponentiation_chain, modulus=gf.characteristic - 1)
            nodes = [result]
            for i, j in chain:
                nodes.append(Multiplication(nodes[i], nodes[j], gf))  # type: ignore
            result = nodes[-1]

            if is_and:
                result = Neg(result, gf).arithmetize("best-effort")

            self._arithmetic_node = result.to_arithmetic()  # TODO: This could be more elegant
            self._is_and = is_and

        assert math.ceil(math.log2(self.breadth)) == self._arithmetic_node.multiplicative_depth()  # type: ignore
        return self._arithmetic_node


def _minimum_cost(operand_count: int, exponentiation_cost: float, p: int) -> float:
    r = math.ceil((p - 1 - operand_count) / (2 - p))
    return r * exponentiation_cost + min(exponentiation_cost, operand_count + r * (2 - p) - 1)


def _find_depth_cost_front(
    operands: Sequence[Tuple[int, float, ArithmeticNode]],
    gf: Type[FieldArray],
    strict_cost_upper: float,
    squaring_cost: float,
    is_and: bool,
) -> CostParetoFront:
    new_operands: List[NaryLogicNode] = [
        InputNaryLogicNode(node, 0 if isinstance(node, Constant) else 2**depth)
        for depth, _, node in operands
    ]

    circuits = minimize_depth_cost(
        new_operands, gf.characteristic, strict_cost_upper, squaring_cost
    )

    front = CostParetoFront(squaring_cost)
    for depth, _, node in circuits:
        front.add(node.to_arithmetic_node(is_and, gf), depth)

    return front


# TODO: This is copied from arbitrary_arithmetic.py
def _generate_sumred_tree(
    operands: Iterable[Tuple[int, InputNaryLogicNode]],
    squaring_cost: float,
) -> Tuple[int, SumReduceNaryLogicNode]:
    queue = [_PrioritizedItem(*operand) for operand in operands]
    heapify(queue)

    while len(queue) > 1:
        a = heappop(queue)
        b = heappop(queue)

        depth = max(a.priority, b.priority) + 1
        heappush(
            queue,
            _PrioritizedItem(
                depth,
                SumReduceNaryLogicNode([a.item, b.item], 2, squaring_cost, [(1, 1)], 2**depth),
            ),
        )

    return (queue[0].priority, queue[0].item)


def minimize_depth_cost(
    operands: List[NaryLogicNode], p: int, strict_cost_upper: float, squaring_cost: float
) -> List[Tuple[int, float, NaryLogicNode]]:
    """Finds the depth-cost Pareto front.

    Returns:
        A front in the form of a list of tuples containing (depth, cost, node).
    """
    assert len(operands) >= 2

    if p == 2:
        result = ProductNaryLogicNode(
            operands, breadth=sum(operand.breadth for operand in operands)
        )
        return [(math.ceil(math.log2(result.breadth)), result.cost, result)]

    if p == 3:
        depth, result = _generate_sumred_tree([(math.ceil(math.log2(operand.breadth)), operand) for operand in operands], squaring_cost)  # type: ignore
        return [(depth, result.cost, result)]

    sorted_operands = sorted(operands, key=lambda op: op.breadth, reverse=True)
    depth_limit = math.ceil(math.log2(sorted_operands[0].breadth))  # + 1

    front = gen_pareto_front(p - 1, p, squaring_cost)
    exponentiation_specs = [
        (depth, chain_cost(chain, squaring_cost), chain) for depth, chain in front
    ]
    _, cheapest_exponentiation_cost, _ = exponentiation_specs[-1]

    mincost = _minimum_cost(len(sorted_operands), cheapest_exponentiation_cost, p)

    circuits = []
    while True:
        breadth_limit = 2**depth_limit
        result = minimize_depth_cost_recursive(
            sorted_operands, breadth_limit, exponentiation_specs, p, strict_cost_upper
        )

        if result is None:
            depth_limit += 1
            continue

        assert result.cost >= mincost
        assert result.cost < strict_cost_upper, f"{result.cost} >= {strict_cost_upper}"

        if result.cost == mincost:
            circuits.append((depth_limit, result.cost, result))
            return circuits

        circuits.append((depth_limit, result.cost, result))
        strict_cost_upper = result.cost

        # TODO: If we want to return the minimum breadth we have to increment at a higher resolution
        depth_limit += 1


def _find_index_breadth(sorted_operands: List[NaryLogicNode], greater_or_equal_to: int) -> int:
    for i in range(len(sorted_operands)):
        if sorted_operands[i].breadth < greater_or_equal_to:
            return i

    return len(sorted_operands)


def _insert(sorted_operands: List[NaryLogicNode], node: NaryLogicNode):
    for i in range(len(sorted_operands)):
        if sorted_operands[i].breadth < node.breadth:
            sorted_operands.insert(i, node)
            return

    sorted_operands.append(node)


def minimize_depth_cost_recursive(  # noqa: PLR0912, PLR0914, PLR0915
    sorted_operands: List[NaryLogicNode],
    breadth_limit: int,
    exponentiation_specs: List[Tuple[int, float, List[Tuple[int, int]]]],
    p: int,
    strict_cost_upper: float,
) -> Optional[NaryLogicNode]:
    """Find a minimum-depth circuit for the given `breadth_limit` and `strict_cost_upper` bound.
    
    Operands must be sorted from deep to shallow.
    Returns the lowest-cost circuit for the given depth.
    The exponentiation_specs must be sorted from high-cost to low-cost.

    Returns:
        A minimum-depth circuit in the form of an `NaryLogicNode` satisfying the constraints, or None if the constraints cannot be satisfied.
    """
    if len(sorted_operands) == 1:
        if breadth_limit >= sorted_operands[0].breadth and strict_cost_upper > 0:
            assert sorted_operands[0].cost < strict_cost_upper
            return sorted_operands[0]
        return None

    # If the breadth limit is exceeded, stop
    if breadth_limit < 1:
        return None

    # If the cost limit is exceeded, stop
    if strict_cost_upper <= 0:
        return None

    # If the lower bound for the cost exceeds the limit, also stop
    _, cheapest_exponentiation_cost, _ = exponentiation_specs[-1]
    lower_bound_cost = _minimum_cost(len(sorted_operands), cheapest_exponentiation_cost, p)
    if lower_bound_cost >= strict_cost_upper:
        return None

    output = None

    for exponentiation_depth, exponentiation_cost, exponentiation_chain in exponentiation_specs:
        # We do not call .cost() in this algorithm because we only consider the cost of the AND/OR subcircuit

        type_2_limit = 2 ** (math.ceil(math.log2(breadth_limit)) - exponentiation_depth)
        if len(sorted_operands) < p:
            if (
                all(operand.breadth <= type_2_limit for operand in sorted_operands)
                and exponentiation_cost < strict_cost_upper
            ):
                # Use a type-2 arithmetization
                depth = math.ceil(math.log2(sorted_operands[0].breadth)) + exponentiation_depth
                output = SumReduceNaryLogicNode(
                    sorted_operands,
                    exponentiation_depth,
                    exponentiation_cost,
                    exponentiation_chain,
                    breadth=2**depth,
                )
                strict_cost_upper = exponentiation_cost

            if (tot := sum(op.breadth for op in sorted_operands)) <= breadth_limit and len(
                sorted_operands
            ) - 1 < strict_cost_upper:
                output = ProductNaryLogicNode(sorted_operands, breadth=tot)
                strict_cost_upper = len(sorted_operands) - 1

            continue

        # At this point, we know that len(sorted_operands) >= p

        # Try a type-1 arithmetization, so no type-2 at all
        if (tot := sum(op.breadth for op in sorted_operands)) <= breadth_limit and len(
            sorted_operands
        ) - 1 < strict_cost_upper:
            output = ProductNaryLogicNode(sorted_operands, breadth=tot)
            strict_cost_upper = len(sorted_operands) - 1

        reduced = all(operand.breadth <= type_2_limit for operand in sorted_operands)
        if reduced:
            if exponentiation_cost >= strict_cost_upper:
                continue

            # Use a type-2 arithmetization on operands of decreasing depth
            cache = set()
            for i in range(len(sorted_operands) - 1):
                selected_operands = sorted_operands[i : (i + p - 1)]
                breadths = tuple(operand.breadth for operand in selected_operands)
                if breadths in cache:
                    continue
                cache.add(breadths)

                depth = math.ceil(math.log2(selected_operands[0].breadth)) + exponentiation_depth

                new_operands = sorted_operands[:i]
                if i + p - 1 < len(sorted_operands):
                    new_operands += sorted_operands[i + p - 1 :]

                breadth = 2**depth
                sum_red = SumReduceNaryLogicNode(
                    sorted_operands[i : i + p - 1],
                    exponentiation_depth,
                    exponentiation_cost,
                    exponentiation_chain,
                    breadth=breadth,
                )
                _insert(new_operands, sum_red)

                potential_output = minimize_depth_cost_recursive(
                    new_operands,
                    breadth_limit,
                    exponentiation_specs,
                    p,
                    strict_cost_upper - exponentiation_cost,
                )
                if potential_output is not None:
                    output = potential_output
                    strict_cost_upper -= potential_output.local_cost()
        else:
            # Isolate all the operands that cannot use a type-2 arithmetization
            first_small_index = _find_index_breadth(sorted_operands, type_2_limit)
            large_operands = sorted_operands[:first_small_index]
            small_operands = sorted_operands[first_small_index:]

            # If there are no small operands, then this arithmetization is not possible
            if len(small_operands) == 0:
                continue

            # Use a type-1 arithmetization for large_operands
            assert len(large_operands) > 0
            cost = len(
                large_operands
            )  # Not -1 because we also need a multiplication with the AND/OR of small_operands
            if cost >= strict_cost_upper:
                continue

            breadth = sum(operand.breadth for operand in large_operands)
            new_breadth_limit = breadth_limit - breadth

            sub_output = minimize_depth_cost_recursive(
                small_operands, new_breadth_limit, exponentiation_specs, p, strict_cost_upper - cost
            )
            if sub_output is not None:
                output = ProductNaryLogicNode(
                    [*large_operands, sub_output], breadth=breadth + sub_output.breadth
                )
                strict_cost_upper -= output.local_cost()

    return output


def all_(*operands: Node) -> And:
    """Returns an `And` node that evaluates to true if any of the given `operands` evaluates to true."""
    assert len(operands) > 0
    return And(set(UnoverloadedWrapper(operand) for operand in operands), operands[0]._gf)
