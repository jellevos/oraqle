

# TODO: Reduce code duplication with ParetoFront
class MultAutParetoFront:
    """
    This represents a Pareto front across four objectives:
    - Multiplicative depth
    - Multiplicative cost
    - Automorphism count
    - Automorphism operand count
    """


class PolyRingPow2:

    def __init__(self, poly_degree_N: int, plaintext_mod_p: int) -> None:
        self._poly_degree_N = poly_degree_N
        self._plaintext_mod_p = plaintext_mod_p
        self._slot_degree = None  # TODO: Call Rust function

    # TODO: Consider creating a property instead for the slot algebra (a Type[FieldArray] for now)
    @property
    def slot_degree(self) -> int:
        return self._slot_degree
    

# class SlotRingElement:
#     pass


# class PolyRingPow2Element:

#     def __init__(self, ) -> None:
#         self._poly_degree_N = poly_degree_N
#         self._plaintext_mod_p = plaintext_mod_p
#         self._slot_degree = None  # TODO: Call Rust function

#     @property
#     def slot_degree(self) -> int:
#         return self._slot_degree


class ZpxNode:
    
    def __init__(self) -> None:
        pass


class GaloisArithmeticNode(ZpxNode):
    pass
