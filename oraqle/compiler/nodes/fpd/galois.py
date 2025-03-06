from typing import Type

from galois import FieldArray
from oraqle.compiler.nodes.fp.univariate import UnivariateNode
from oraqle.compiler.nodes.fpd.abstract import FpdNode


# TODO: This should be a univariate FpNode
class FieldNorm(UnivariateNode):
    
    def __init__(self, node: FpdNode, gf: Type[FieldArray]) -> None:
        # TODO: Implement the arithmetization
        # TODO: Allow UnivariateNode to take generic Nodes (e.g. FpdNode) as input?
        super().__init__(node, gf)


# TODO: Trace implementation
