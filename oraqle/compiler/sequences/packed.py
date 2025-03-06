

# TODO: Consider renaming packed to SIMD

class PackedSequence:
    """
    Sequences are 0-indexed. We currently only support power-of-two cyclotomics, so all values (length, stride, offset) must be powers of two.
    
    Such a sequence is made up of elements in F_{p^d}, where d is the degree of a slot. See `PackedElement`.
    """

    def __init__(self, length: int, stride: int, offset: int) -> None:
        assert (length & (length - 1)) == 0, "Length should be a power of two"
        assert (stride & (stride - 1)) == 0, "Stride should be a power of two"
        assert (offset & (offset - 1)) == 0, "Offset should be a power of two"
        self._length = length
        self._stride = stride
        self._offset = offset


# TODO: Remove below, FpdNodes are is isomorphic to the slot ring 

# class PackedElement:
#     """
#     An element of F_{p^d}, so it is isomorphic to the slot ring.
#     """
    
#     def __sub__(self, other: "PackedElement") -> Sub:
#         # TODO: Create subtraction node


# TODO: Convert many inputs representing PackedElements to few inputs representing PackedCiphertexts
