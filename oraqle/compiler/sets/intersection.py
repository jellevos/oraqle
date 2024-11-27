from oraqle.compiler.circuit import ArithmeticCircuit
from oraqle.compiler.nodes.abstract import CostParetoFront, Node, UnoverloadedWrapper
from oraqle.compiler.nodes.flexible import CommutativeUniqueReducibleNode
from oraqle.compiler.sets.bitset import BitSet, BitSetIntersection
from oraqle.compiler.sets.set import AbstractSet, Set

from galois import GF, FieldArray


# TODO: Should we check if the sets are from the same universe?
class Intersection(CommutativeUniqueReducibleNode[AbstractSet]):
    
    @property
    def _hash_name(self) -> str:
        return "intersection"

    @property
    def _node_label(self) -> str:
        return "Intersection"

    def _arithmetize_inner(self, strategy: str) -> Node:
        # TODO: This should arithmetize differently when it is the output of the circuit (or it can be leaked)
        # TODO: In the future, this should try all possible arithmetizations for intersections. How do we prioritize? Do we return multiple results?
        assert all(isinstance(s.node, (Set, BitSet)) for s in self._operands)
        return BitSetIntersection({UnoverloadedWrapper(BitSet.coerce_from(s.node)) for s in self._operands}, self._gf).arithmetize(strategy)  # type: ignore
    
    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        raise NotImplementedError()
    
    def _inner_operation(self, a: FieldArray, b: FieldArray) -> FieldArray:
        # TODO: This should take sets as input
        raise NotImplementedError()
    

def intersection(*operands: AbstractSet) -> Intersection:
    """Returns an `Intersection` node that computes the intersection of the input sets."""
    assert len(operands) > 0
    return Intersection(set(UnoverloadedWrapper(operand) for operand in operands), operands[0]._gf)



if __name__ == '__main__':
    gf = GF(11)
    sets = {UnoverloadedWrapper(Set(f"s{i}", gf, 100)) for i in range(3)}
    
    result = Intersection(sets, gf)

    arithmetization = result.arithmetize("best-effort")
    print(arithmetization)
    ArithmeticCircuit([arithmetization]).to_pdf("arithmetization.pdf")
