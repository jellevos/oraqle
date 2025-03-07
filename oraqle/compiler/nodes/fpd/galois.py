from typing import Optional, Type

from galois import FieldArray
from oraqle.compiler.nodes.fp.abstract import CostParetoFront
from oraqle.compiler.nodes.fp.fixed import ArithmeticNode
from oraqle.compiler.nodes.fp.univariate import UnivariateFpNode
from oraqle.compiler.nodes.fpd.abstract import FpdNode


class FieldNorm(UnivariateFpNode[FpdNode]):

    @property
    def _node_label(self) -> str:
        return "Norm"
    
    @property
    def _hash_name(self) -> str:
        return f"field_norm_{self._norm_degree}"  # TODO: Should we include gf in the hash?
    
    @property
    def _node_shape(self) -> str:
        return "box"
    
    def __init__(self, node: FpdNode, norm_degree: Optional[int] = None) -> None:
        # TODO: Implement the arithmetization
        super().__init__(node, node._gf)
        self._norm_degree = self._gf.degree if norm_degree is None else norm_degree

    def _arithmetize_inner(self, strategy: str) -> ArithmeticNode:
        chain up to norm_degree
    
    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        raise NotImplementedError("TODO")
    
    def _operation_inner(self, input: FieldArray) -> FieldArray:
        return input.field_norm()


# TODO: Trace implementation
