"""Module containing fixed nodes: nodes with a fixed number of inputs."""
from abc import abstractmethod
from typing import Callable, Dict, List

from galois import FieldArray

from oraqle.compiler.nodes.abstract import CostParetoFront, Node


class FixedNode(Node):
    """A node with a fixed number of operands."""

    @abstractmethod
    def operands(self) -> List["Node"]:
        """Returns the operands (children) of this node. The list can be empty."""

    @abstractmethod
    def set_operands(self, operands: List["Node"]):
        """Overwrites the operands of this node."""
        # TODO: Consider replacing this method with a graph traversal method that applies a function on all operands and replaces them.

    
    def apply_function_to_operands(self, function: Callable[[Node], None]):  # noqa: D102
        for operand in self.operands():
            function(operand)

    
    def replace_operands_using_function(self, function: Callable[[Node], Node]):  # noqa: D102
        self.set_operands([function(operand) for operand in self.operands()])
        # TODO: These caches should only be cleared if this is an ArithmeticNode
        self._multiplications = None
        self._squarings = None
        self._depth_cache = None

    
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
    
    def arithmetize(self, strategy: str) -> "Node":  # noqa: D102
        if self._arithmetize_cache is None:
            if self._arithmetize_depth_cache is not None:
                return self._arithmetize_depth_cache.get_lowest_value()  # type: ignore

            # If we know all operands we can simply evaluate this node
            operands = self.operands()
            if len(operands) > 0 and all(
                hasattr(operand, "_value") for operand in operands
            ):  # This is a hacky way of checking whether the operands are all constant
                from oraqle.compiler.nodes.leafs import Constant

                self._arithmetize_cache = Constant(self.operation([operand._value for operand in self.operands()]))  # type: ignore
            else:
                self._arithmetize_cache = self._arithmetize_inner(strategy)

        return self._arithmetize_cache

    @abstractmethod
    def _arithmetize_inner(self, strategy: str) -> "Node":
        pass

    # TODO: Reduce code duplication
    
    def arithmetize_depth_aware(self, cost_of_squaring: float) -> CostParetoFront:  # noqa: D102
        if self._arithmetize_depth_cache is None:
            if self._arithmetize_cache is not None:
                raise Exception("This should not happen")

            # If we know all operands we can simply evaluate this node
            operands = self.operands()
            if len(operands) > 0 and all(
                hasattr(operand, "_value") for operand in operands
            ):  # This is a hacky way of checking whether the operands are all constant
                from oraqle.compiler.nodes.leafs import Constant

                self._arithmetize_depth_cache = CostParetoFront.from_leaf(Constant(self.operation([operand._value for operand in self.operands()])), cost_of_squaring)  # type: ignore
            else:
                self._arithmetize_depth_cache = self._arithmetize_depth_aware_inner(
                    cost_of_squaring
                )

        assert self._arithmetize_depth_cache is not None
        return self._arithmetize_depth_cache

    @abstractmethod
    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        pass


class BinaryNode(FixedNode):
    """A node with two operands."""
