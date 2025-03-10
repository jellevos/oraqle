from abc import abstractmethod
from typing import List, Type

from galois import FieldArray
from oraqle.compiler.nodes.abstract import Node
from oraqle.compiler.nodes.fpd.fixed import FixedFpdNode
from oraqle.compiler.nodes.univariate import UnivariateNode


class UnivariateFpdNode[Operand: Node](UnivariateNode[Operand], FixedFpdNode[Operand]):
    """An abstract node with a single FpNode input."""

    def __init__(self, node: Operand, gf: Type[FieldArray]):
        """Initialize a univariate node."""
        self._node = node
        # TODO: assert not isinstance(node, FpdConstant)
        FixedFpdNode.__init__(self, gf)
        UnivariateNode.__init__(self, node)

    @abstractmethod
    def _operation_inner(self, input: FieldArray) -> FieldArray:
        """Evaluate the operation on the input. This method does not have to cache."""

    def operation(self, operands: List[FieldArray]) -> FieldArray:  # noqa: D102
        return self._operation_inner(operands[0])
