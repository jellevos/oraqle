from oraqle.compiler.boolean.bool import Boolean
from oraqle.compiler.boolean.bool_and import And
from oraqle.compiler.boolean.bool_or import Or
from oraqle.compiler.nodes.abstract import Node
from oraqle.compiler.nodes.arbitrary_arithmetic import sum_


class UnreducedBool(Boolean):
    pass


class InvUnreducedBool(Boolean):
    pass


class UnreducedOr(Or):

    def _arithmetize_inner(self, strategy: str) -> Node:
        # FIXME: Do we need to randomize (i.e. make it a Sum with random multiplicities?)
        return sum_(*self._operands)
    
    # TODO: Depth-aware


class InvUnreducedAnd(And):

    def _arithmetize_inner(self, strategy: str) -> Node:
        # FIXME: Do we need to randomize (i.e. make it a Sum with random multiplicities?)
        return sum_(*self._operands)

    # TODO: Depth-aware
