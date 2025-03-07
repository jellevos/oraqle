from typing import List, Type

from galois import FieldArray
from oraqle.compiler.nodes.fp.abstract import FpNode
from oraqle.compiler.nodes.fpd.abstract import FpdNode
from oraqle.compiler.nodes.fpd.galois import FieldNorm
from oraqle.compiler.nodes.zpx_cycl_pow2.abstract import MultAutParetoFront, PolyRingPow2, ZpxNode


def poly_eval_galois_within_degree(gf: Type[FieldArray], coefficients: List[int], element: FpdNode) -> FpNode:
    assert (len(coefficients) - 1) <= gf.degree

    added_poly, factor, inverse_factor, poly_degree = None  # TODO: Call Rust code

    # Add a poly and multiply with a factor to ensure the degree is a power of two and the poly is monic
    add poly
    multiply with factor

    # Perform the norm computation
    alpha: FpdNode = None  # TODO: Call Rust code
    res = FieldNorm(alpha - element, poly_degree, poly_ring)

    # Undo the added_poly and the factor
    if inverse_factor != 1:
        res = res * inverse_factor
    if added_poly nonzero:
        res += 

    return res


# TODO: This should probably be a UnivariateZpxNode
class UnivariatePolyZpx(ZpxNode):
    pass
