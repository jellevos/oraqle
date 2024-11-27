"""Module containing leaf nodes: i.e. nodes without an input."""
from typing import Any, Dict, List, Set, Tuple, Type

from galois import FieldArray

from oraqle.compiler.graphviz import DotFile
from oraqle.compiler.instructions import ArithmeticInstruction, InputInstruction
from oraqle.compiler.nodes.abstract import ArithmeticNode, CostParetoFront, Node, select_stack_index
from oraqle.compiler.nodes.fixed import FixedNode


class ArithmeticLeafNode(FixedNode, ArithmeticNode):
    """An ArithmeticLeafNode is an ArithmeticNode with no inputs."""

    def operands(self) -> List[Node]:  # noqa: D102
        return []

    def set_operands(self, operands: List["Node"]):  # noqa: D102
        pass
    
    def _arithmetize_inner(self, strategy: str) -> Node:
        return self

    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        return CostParetoFront.from_leaf(self, cost_of_squaring)
    
    def multiplicative_depth(self) -> int:  # noqa: D102
        return 0

    def multiplicative_size(self) -> int:  # noqa: D102
        return 0
    
    def multiplications(self) -> Set[int]:  # noqa: D102
        return set()
    
    def squarings(self) -> Set[int]:  # noqa: D102
        return set()


# TODO: Merge ArithmeticInput and Input using multiple inheritance
class Input(ArithmeticLeafNode):
    """Represents a named input to the arithmetic circuit."""

    @property
    def _overriden_graphviz_attributes(self) -> dict:
        return {"shape": "circle", "style": "filled", "fillcolor": "lightsteelblue1"}

    @property
    def _hash_name(self) -> str:
        return "input"

    @property
    def _node_label(self) -> str:
        return self._name

    def __init__(self, name: str, gf: Type[FieldArray]) -> None:
        """Initialize an input with the given `name`."""
        super().__init__(gf)
        self._name = name

    
    def operation(self, operands: List[FieldArray]) -> FieldArray:  # noqa: D102
        raise Exception()

    
    def evaluate(self, actual_inputs: Dict[str, FieldArray]) -> FieldArray:  # noqa: D102
        return actual_inputs[self._name]

    
    def to_graph(self, graph_builder: DotFile) -> int:  # noqa: D102
        if self._to_graph_cache is None:
            label = self._name

            self._to_graph_cache = graph_builder.add_node(
                label=label, **self._overriden_graphviz_attributes
            )

        return self._to_graph_cache

    def __hash__(self) -> int:
        return hash(self._name)

    
    def is_equivalent(self, other: Node) -> bool:  # noqa: D102
        if not isinstance(other, self.__class__):
            return False

        return self._name == other._name

    
    def create_instructions(  # noqa: D102
        self,
        instructions: List[ArithmeticInstruction],
        stack_counter: int,
        stack_occupied: List[bool],
    ) -> Tuple[int, int]:
        if self._instruction_cache is None:
            self._instruction_cache = select_stack_index(stack_occupied)
            instructions.append(InputInstruction(self._instruction_cache, self._name))

        return self._instruction_cache, stack_counter


class Constant(ArithmeticLeafNode):
    """Represents a Node with a constant value."""

    @property
    def _overriden_graphviz_attributes(self) -> dict:
        return {"style": "filled", "fillcolor": "grey80", "shape": "circle"}

    @property
    def _hash_name(self) -> str:
        return "constant"

    @property
    def _node_label(self) -> str:
        return str(self._value)

    def __init__(self, value: FieldArray):
        """Initialize a Node with the given `value`."""
        super().__init__(value.__class__)
        self._value = value

    
    def operation(self, operands: List[FieldArray]) -> FieldArray:  # noqa: D102
        return self._value

    
    def to_graph(self, graph_builder: DotFile) -> Any:  # noqa: D102
        if self._to_graph_cache is None:
            label = str(self._value)

            self._to_graph_cache = graph_builder.add_node(
                label=label, **self._overriden_graphviz_attributes
            )

        return self._to_graph_cache

    def __hash__(self) -> int:
        return hash(int(self._value))

    
    def is_equivalent(self, other: Node) -> bool:  # noqa: D102
        if not isinstance(other, self.__class__):
            return False

        return self._value == other._value

    
    def add(self, other: "Node", flatten=True) -> "Node":  # noqa: D102
        if isinstance(other, Constant):
            return Constant(self._value + other._value)

        return other.add(self, flatten)

    
    def mul(self, other: "Node", flatten=True) -> "Node":  # noqa: D102
        if isinstance(other, Constant):
            return Constant(self._value * other._value)

        return other.mul(self, flatten)

    
    def bool_or(self, other: "Node", flatten=True) -> Node:  # noqa: D102
        if isinstance(other, Constant):
            return Constant(self._gf(bool(self._value) | bool(other._value)))

        return other.bool_or(self, flatten)
    
    def bool_and(self, other: "Node", flatten=True) -> Node:  # noqa: D102
        if isinstance(other, Constant):
            return Constant(self._gf(bool(self._value) & bool(other._value)))

        return other.bool_and(self, flatten)
    
    def create_instructions(  # noqa: D102
        self,
        instructions: List[ArithmeticInstruction],
        stack_counter: int,
        stack_occupied: List[bool],
    ) -> Tuple[int]:
        raise NotImplementedError("The circuit is a constant.")


class DummyNode(FixedNode):
    """A DummyNode is a fixed node with no inputs and no behavior."""

    def operands(self) -> List[Node]:  # noqa: D102
        return []

    def set_operands(self, operands: List["Node"]):  # noqa: D102
        pass
