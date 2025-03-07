"""Module containing fixed nodes: nodes with a fixed number of inputs."""
from abc import abstractmethod
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set, Tuple

from galois import FieldArray

from oraqle.compiler.instructions import ArithmeticInstruction
from oraqle.compiler.nodes.abstract import Node
from oraqle.compiler.nodes.fixed import FixedNode
from oraqle.compiler.nodes.fp.abstract import CostParetoFront, FpNode, ParetoFront


class FixedFpNode[Operand: FpNode](FixedNode[Operand], FpNode):
    """A node with a fixed number of operands that are all FpNodes as well."""
    
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
    
    def arithmetize(self, strategy: str) -> "ArithmeticNode":  # noqa: D102
        if self._arithmetize_cache is None:
            if self._arithmetize_depth_cache is not None:
                return self._arithmetize_depth_cache.get_lowest_value()  # type: ignore

            # If we know all operands we can simply evaluate this node
            operands = self.operands()
            if len(operands) > 0 and all(
                hasattr(operand, "_value") for operand in operands
            ):  # This is a hacky way of checking whether the operands are all constant
                from oraqle.compiler.nodes.fp.leafs import Constant

                self._arithmetize_cache = Constant(self.operation([operand._value for operand in self.operands()]))  # type: ignore
            else:
                self._arithmetize_cache = self._arithmetize_inner(strategy)

        return self._arithmetize_cache

    @abstractmethod
    def _arithmetize_inner(self, strategy: str) -> "ArithmeticNode":
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
                from oraqle.compiler.nodes.fp.leafs import Constant

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


class BinaryFpNode(FixedFpNode):
    """A node with two operands."""


class ArithmeticNode(FixedFpNode["ArithmeticNode"]):
    """
    A special type of FixedFpNode that indicates only arithmetic operations. These are the primitives in an arithmetic circuit.
    
    This node is an extension of Node to indicate that this is a node permitted in a purely arithmetic circuit (with binary additions and multiplications).
    The ArithmeticNode 'mixin' must always come before the base class in the class declaration.
    """

    # ArithmeticNode should be like an interface; it should not have an __init__ method.

    def clear_cache(self, already_cleared: Set[int]):
        """Clears any cached values of the node and any of its operands."""
        # FIXME: The cache should not be cleared twice for the same node, but there is no way to check this.
        if id(self) not in already_cleared:
            for node in self.operands():
                node.clear_cache(already_cleared)

        self._evaluate_cache: Optional[FieldArray] = None
        self._to_graph_cache: Optional[int] = None
        self._arithmetize_cache: Optional[ArithmeticNode] = None
        self._arithmetize_depth_cache: Optional[ParetoFront] = None
        self._instruction_cache: Optional[int] = None
        self._arithmetic_cache: Optional[ArithmeticNode] = None
        self._parent_count_cache: Optional[int] = None

        self._hash = None

        already_cleared.add(id(self))

    @abstractmethod
    def multiplicative_depth(self) -> int:
        """Computes the multiplicative depth of this node and its children recursively.
        
        Returns:
        The largest number of multiplications from the output of this node to the leafs of this subcircuit.
        """

    def multiplicative_size(self) -> int:
        """Computes the multiplicative size (number of multiplications) by counting the size of the set returned by self.multiplications().
        
        Returns:
        The number of multiplications in this subcircuit.
        """
        return len(self.multiplications())

    def multiplicative_cost(self, cost_of_squaring: float) -> float:
        """Computes the multiplicative cost (number of general multiplications + cost_of_squaring * squarings).
        
        It does so by counting the size of the sets returned by self.multiplications() and self.squarings().

        Returns:
            The number of proper multiplications + the cost of squaring * the number of squarings.
        """
        return (
            len(self.multiplications())
            - len(self.squarings())
            + cost_of_squaring * len(self.squarings())
        )

    @abstractmethod
    def multiplications(self) -> Set[int]:
        """Returns a set of all the multiplications in this tree of descendants, including itself.
        
        This includes any squarings.
        """

    @abstractmethod
    def squarings(self) -> Set[int]:
        """Returns a set of all the squarings in this tree of descendants, including itself."""

    @abstractmethod
    def create_instructions(
        self,
        instructions: List[ArithmeticInstruction],
        stack_counter: int,
        stack_occupied: List[bool],
    ) -> Tuple[int, int]:
        """Creates a set of instructions of this node to the given file. Returns the index in the stack of the output and the stack_counter.
        
        !!! note
            This method assumes that the _parent_count of each node is up to date.
        """

    def to_arithmetic(self) -> "ArithmeticNode":  # noqa: D102
        return self
