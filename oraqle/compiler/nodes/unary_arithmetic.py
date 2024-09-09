"""This module contains `ArithmeticNode`s with a single input: Constant additions and constant multiplications."""
from typing import List, Optional, Set, Tuple

from galois import FieldArray

from oraqle.compiler.graphviz import DotFile
from oraqle.compiler.instructions import (
    ArithmeticInstruction,
    ConstantAdditionInstruction,
    ConstantMultiplicationInstruction,
)
from oraqle.compiler.nodes.abstract import ArithmeticNode, CostParetoFront, Node, select_stack_index
from oraqle.compiler.nodes.univariate import UnivariateNode

# TODO: There is (going to be) a lot of code duplication between these two classes


class ConstantAddition(UnivariateNode, ArithmeticNode):
    """This node represents a multiplication of another node with a constant."""

    @property
    def _overriden_graphviz_attributes(self) -> dict:
        return {"style": "rounded,filled", "fillcolor": "grey80"}

    @property
    def _node_shape(self) -> str:
        return "square"

    @property
    def _hash_name(self) -> str:
        return f"constant_add_{self._constant}"

    @property
    def _node_label(self) -> str:
        return "+"

    def __init__(self, node: ArithmeticNode, constant: FieldArray):
        """Represents the operation `constant + node`."""
        super().__init__(node, constant.__class__)
        self._constant = constant
        assert constant != 0

        self._depth_cache: Optional[int] = None

    
    def _operation_inner(self, input: FieldArray) -> FieldArray:
        return input + self._constant

    
    def multiplicative_depth(self) -> int:  # noqa: D102
        if self._depth_cache is None:
            self._depth_cache = self._node.multiplicative_depth()

        return self._depth_cache

    
    def multiplications(self) -> Set[int]:  # noqa: D102
        return self._node.multiplications()

    
    def squarings(self) -> Set[int]:  # noqa: D102
        return self._node.squarings()

    
    def create_instructions(  # noqa: D102
        self,
        instructions: List[ArithmeticInstruction],
        stack_counter: int,
        stack_occupied: List[bool],
    ) -> Tuple[int, int]:
        self._node: ArithmeticNode

        if self._instruction_cache is None:
            operand_index, stack_counter = self._node.create_instructions(
                instructions, stack_counter, stack_occupied
            )

            self._node._parent_count -= 1
            if self._node._parent_count == 0:
                stack_occupied[self._node._instruction_cache] = False  # type: ignore

            self._instruction_cache = select_stack_index(stack_occupied)

            instructions.append(
                ConstantAdditionInstruction(self._instruction_cache, operand_index, self._constant)
            )

        return self._instruction_cache, stack_counter

    
    def _arithmetize_inner(self, strategy: str) -> Node:
        return self

    
    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        front = CostParetoFront(cost_of_squaring)
        for _, _, node in self._node.arithmetize_depth_aware(cost_of_squaring):
            front.add(ConstantAddition(node, self._constant))
        return front

    
    def to_graph(self, graph_builder: DotFile) -> int:  # noqa: D102
        if self._to_graph_cache is None:
            super().to_graph(graph_builder)
            self._to_graph_cache: int

            # TODO: Add known_by
            graph_builder.add_link(
                graph_builder.add_node(
                    label=str(self._constant), shape="circle", style="filled", fillcolor="grey92"
                ),
                self._to_graph_cache,
            )

        return self._to_graph_cache


class ConstantMultiplication(UnivariateNode, ArithmeticNode):
    """This node represents a multiplication of another node with a constant."""

    @property
    def _overriden_graphviz_attributes(self) -> dict:
        return {"style": "rounded,filled", "fillcolor": "grey80"}

    @property
    def _node_shape(self) -> str:
        return "square"

    @property
    def _hash_name(self) -> str:
        return f"constant_mul_{self._constant}"

    @property
    def _node_label(self) -> str:
        return "×"  # noqa: RUF001

    def __init__(self, node: Node, constant: FieldArray):
        """Represents the operation `constant * node`."""
        super().__init__(node, constant.__class__)
        self._constant = constant
        assert constant != 0
        assert constant != 1

        self._depth_cache: Optional[int] = None

    def _operation_inner(self, input: FieldArray) -> FieldArray:
        return input * self._constant  # type: ignore

    
    def multiplicative_depth(self) -> int:  # noqa: D102
        if self._depth_cache is None:
            self._depth_cache = self._node.multiplicative_depth()  # type: ignore

        return self._depth_cache  # type: ignore

    
    def multiplications(self) -> Set[int]:  # noqa: D102
        return self._node.multiplications()  # type: ignore

    
    def squarings(self) -> Set[int]:  # noqa: D102
        return self._node.squarings()  # type: ignore

    
    def create_instructions(  # noqa: D102
        self,
        instructions: List[ArithmeticInstruction],
        stack_counter: int,
        stack_occupied: List[bool],
    ) -> Tuple[int, int]:
        self._node: ArithmeticNode

        if self._instruction_cache is None:
            operand_index, stack_counter = self._node.create_instructions(
                instructions, stack_counter, stack_occupied
            )

            self._node._parent_count -= 1
            if self._node._parent_count == 0:
                stack_occupied[self._node._instruction_cache] = False  # type: ignore

            self._instruction_cache = select_stack_index(stack_occupied)

            instructions.append(
                ConstantMultiplicationInstruction(
                    self._instruction_cache, operand_index, self._constant
                )
            )

        return self._instruction_cache, stack_counter

    
    def _arithmetize_inner(self, strategy: str) -> Node:
        return self

    
    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        front = CostParetoFront(cost_of_squaring)
        for _, _, node in self._node.arithmetize_depth_aware(cost_of_squaring):
            front.add(ConstantMultiplication(node, self._constant))
        return front

    
    def to_graph(self, graph_builder: DotFile) -> int:  # noqa: D102
        if self._to_graph_cache is None:
            super().to_graph(graph_builder)
            self._to_graph_cache: int

            # TODO: Add known_by
            graph_builder.add_link(
                graph_builder.add_node(
                    label=str(self._constant), shape="circle", style="filled", fillcolor="grey92"
                ),
                self._to_graph_cache,
            )

        return self._to_graph_cache
