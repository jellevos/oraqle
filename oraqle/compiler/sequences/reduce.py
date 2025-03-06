from typing import Type
from galois import FieldArray
from oraqle.compiler.nodes.fp.abstract import Node
from oraqle.compiler.nodes.fp.leafs import Input
from oraqle.compiler.sequences.map import Map
from oraqle.compiler.sequences.packed import PackedSequence


class Reduce(Node):

    def __init__(self, sequence: PackedSequence, operation: Node, gf: type[FieldArray]):
        super().__init__(gf)


# TODO: Consider implementing this example
# if __name__ == "__main__":
#     a = Input()  # TODO: Make packed ZpInput
#     b = Input()  # TODO: Make packed ZpInput
#     # TODO: Somehow, zip
#     # Dot product
#     res = Reduce(Map(ab, *), +)

#     ac = res.arithmetize("best-effort", plaintext_algebra)
#     # TODO: The inputs in ac should be SimdInputs now (with allocations)
#     # TODO: if inputs are short enough, then it's an multiplication between the two SimdInputs, then rots and adds
#     # I think, because SimdInputs are subclasses of ZpxNode, we do not have to impl Add and Mul for it (it's derived)
