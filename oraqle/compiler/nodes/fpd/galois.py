from typing import List, Optional, Self, Set, Tuple, Type

from galois import FieldArray
from oraqle.compiler.instructions import ArithmeticInstruction
from oraqle.compiler.nodes.fp.abstract import CostParetoFront
from oraqle.compiler.nodes.fp.fixed import ArithmeticNode, GaloisArithmeticNode
from oraqle.compiler.nodes.fp.univariate import UnivariateFpNode
from oraqle.compiler.nodes.fpd.abstract import FpdNode
from oraqle.compiler.nodes.fpd.univariate import UnivariateFpdNode


class FrobeniusAutomorphism(UnivariateFpdNode[GaloisArithmeticNode], GaloisArithmeticNode):

    @property
    def _node_label(self) -> str:
        return f"Aut p^{self._power}"
    
    @property
    def _hash_name(self) -> str:
        return f"aut_{self._power}"  # FIXME: Include the gf in the hash?

    @property
    def _node_shape(self) -> str:
        return "box"

    def __init__(self, node: GaloisArithmeticNode, power: int):
        self._power = power
        super().__init__(node, node._gf)

    def _arithmetize_inner(self, strategy: str) -> ArithmeticNode:
        raise NotImplementedError("This automorphism is not an arithmetic node")
    
    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        raise NotImplementedError("This automorphism is not an arithmetic node")
    
    def _galois_arithmetize_inner(self) -> GaloisArithmeticNode:
        return self
    
    def multiplicative_depth(self) -> int:
        return self._node.multiplicative_depth()
    
    def multiplications(self) -> Set[int]:
        return self._node.multiplications()
    
    def squarings(self) -> Set[int]:
        return self._node.squarings()
    
    def create_instructions(self, instructions: List[ArithmeticInstruction], stack_counter: int, stack_occupied: List[bool]) -> Tuple[int, int]:
        raise NotImplementedError("TODO")
    
    def _operation_inner(self, input: FieldArray) -> FieldArray:
        return input ** self._gf.characteristic


class FieldNorm(UnivariateFpNode[FpdNode]):

    @property
    def _node_label(self) -> str:
        return "Norm"
    
    @property
    def _hash_name(self) -> str:
        return f"field_norm"  # TODO: Should we include gf in the hash?
    
    @property
    def _node_shape(self) -> str:
        return "box"
    
    def __init__(self, node: FpdNode) -> None:
        # TODO: Implement the arithmetization
        assert ((node._gf.degree - 1) & node._gf.degree) == 0, "For now, the degree must be a power of two to compute the norm"
        super().__init__(node, node._gf)

    def _arithmetize_inner(self, strategy: str) -> ArithmeticNode:
        # We can use exponentiation without automorphisms, it's just less efficient.
        raise NotImplementedError("TODO")
    
    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        raise NotImplementedError("TODO")
    
    def _galois_arithmetize_inner(self) -> GaloisArithmeticNode:
        # TODO: Implement the HElib strategy, which allows degrees beyond powers of two
        res = self._node.galois_arithmetize_fpd()
        for i in range(self._gf.degree):
            res = FrobeniusAutomorphism(res, 1 << i) * res
        return res

    def _operation_inner(self, input: FieldArray) -> FieldArray:
        return input.field_norm()


# TODO: Trace implementation
