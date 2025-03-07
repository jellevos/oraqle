from typing import Type

from galois import FieldArray
from oraqle.compiler.nodes.fp.univariate import UnivariateFpNode
from oraqle.compiler.nodes.fpd.abstract import FpdNode


class FieldNorm(UnivariateFpNode[FpdNode]):
    
    def __init__(self, node: FpdNode, gf: Type[FieldArray]) -> None:
        # TODO: Implement the arithmetization
        super().__init__(node, gf)


# TODO: Trace implementation
