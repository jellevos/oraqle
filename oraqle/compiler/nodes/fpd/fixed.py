# from typing import Type

# from galois import FieldArray
# from oraqle.compiler.nodes.fp.fixed import FixedFpNode
# from oraqle.compiler.nodes.fpd.abstract import FpdNode
# from oraqle.compiler.nodes.univariate import UnivariateNode


# class UnivariateFpdNode[Operand: FpdNode](UnivariateNode[Operand], FixedFpdNode[Operand]):
#     """An abstract node with a single FpNode input."""

#     def __init__(self, node: FpdNode, gf: Type[FieldArray]):
#         """Initialize a univariate node."""
#         self._node = node
#         # TODO: assert not isinstance(node, FpdConstant)
#         FixedFpNode.__init__(self, gf)
#         UnivariateNode.__init__(self, node)


from abc import abstractmethod
from typing import Type, override

from galois import FieldArray
from oraqle.compiler.nodes.abstract import Node
from oraqle.compiler.nodes.fixed import FixedNode
from oraqle.compiler.nodes.fp.abstract import CostParetoFront
from oraqle.compiler.nodes.fp.fixed import ArithmeticNode, GaloisArithmeticNode
from oraqle.compiler.nodes.fpd.abstract import FpdNode


class FixedFpdNode[Operand: Node](FixedNode[Operand], FpdNode):
    """A node with a fixed number of operands."""
    
    @override
    def arithmetize_fpd(self, strategy: str, circuit_gf: Type[FieldArray]) -> "ArithmeticNode":  # noqa: D102
        if self._arithmetize_cache is None:
            if self._arithmetize_depth_cache is not None:
                return self._arithmetize_depth_cache.get_lowest_value()  # type: ignore

            # If we know all operands we can simply evaluate this node
            operands = self.operands()
            if len(operands) > 0 and all(
                hasattr(operand, "_value") for operand in operands
            ):  # This is a hacky way of checking whether the operands are all constant
                from oraqle.compiler.nodes.fp.leafs import FpConstant

                self._arithmetize_cache = FpConstant(self.operation([operand._value for operand in self.operands()]))  # type: ignore
            else:
                self._arithmetize_cache = self._arithmetize_inner(strategy, circuit_gf)

        return self._arithmetize_cache

    @abstractmethod
    def _arithmetize_inner(self, strategy: str, circuit_gf: Type[FieldArray]) -> "ArithmeticNode":
        pass

    # TODO: Reduce code duplication
    
    @override
    def arithmetize_depth_aware(self, cost_of_squaring: float) -> CostParetoFront:  # noqa: D102
        if self._arithmetize_depth_cache is None:
            if self._arithmetize_cache is not None:
                raise Exception("This should not happen")

            # If we know all operands we can simply evaluate this node
            operands = self.operands()
            if len(operands) > 0 and all(
                hasattr(operand, "_value") for operand in operands
            ):  # This is a hacky way of checking whether the operands are all constant
                from oraqle.compiler.nodes.fp.leafs import FpConstant

                self._arithmetize_depth_cache = CostParetoFront.from_leaf(FpConstant(self.operation([operand._value for operand in self.operands()])), cost_of_squaring)  # type: ignore
            else:
                self._arithmetize_depth_cache = self._arithmetize_depth_aware_inner(
                    cost_of_squaring
                )

        assert self._arithmetize_depth_cache is not None
        return self._arithmetize_depth_cache

    @abstractmethod
    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        pass

    @override
    def galois_arithmetize_fpd(self, circuit_gf: Type[FieldArray]) -> GaloisArithmeticNode:
        if self._galois_arithmetize_cache is None:
            self._galois_arithmetize_cache = self._galois_arithmetize_inner(circuit_gf)
        
        return self._galois_arithmetize_cache
    
    def _galois_arithmetize_inner(self, circuit_gf: Type[FieldArray]) -> GaloisArithmeticNode:
        return self._arithmetize_inner("best-effort", circuit_gf)
