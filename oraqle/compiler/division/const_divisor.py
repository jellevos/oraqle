import math
from galois import GF, FieldArray
from oraqle.add_chains.addition_chains_front import gen_pareto_front
from oraqle.add_chains.solving import extract_indices
from oraqle.compiler.comparison.in_upper_half import mod_pow
from oraqle.compiler.division.remainder import Remainder
from oraqle.compiler.nodes.abstract import CostParetoFront, Node
from oraqle.compiler.nodes.binary_arithmetic import Addition, Multiplication
from oraqle.compiler.nodes.leafs import Input
from oraqle.compiler.nodes.unary_arithmetic import ConstantMultiplication
from oraqle.compiler.nodes.univariate import UnivariateNode


class DivideBy(UnivariateNode):

    @property
    def _node_shape(self) -> str:
        return "box"

    @property
    def _hash_name(self) -> str:
        return f"modulo_{self._divisor}"

    @property
    def _node_label(self) -> str:
        return f"/ {self._divisor}"

    def __init__(self, node: Node, divisor: int):
        self._divisor = divisor
        super().__init__(node, node._gf)

    def _arithmetize_inner(self, strategy: str) -> Node:
        raise NotImplementedError("TODO!")
    
    def _operation_inner(self, input: FieldArray) -> FieldArray:
        return self._gf(round(int(input) / self._divisor))

    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        correction = self._divisor // 2
        shifted = self._node + correction
        quantized = shifted - Remainder(shifted, self._divisor)
        p = self._gf.characteristic
        div_inv = mod_pow(self._divisor, p - 2, p)
        return (quantized * div_inv).arithmetize_depth_aware(cost_of_squaring)
