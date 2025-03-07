from abc import abstractmethod
from typing import Callable, Dict, List
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

    def apply_function_to_operands(self, function: Callable[[Operand], None]):  # noqa: D102
        for operand in self.operands():
            function(operand)
    
    def replace_operands_using_function(self, function: Callable[[Operand], Operand]):  # noqa: D102
        self.set_operands([function(operand) for operand in self.operands()])
        # TODO: These caches should only be cleared if this is an ArithmeticNode
        self._multiplications = None
        self._squarings = None
        self._depth_cache = None
