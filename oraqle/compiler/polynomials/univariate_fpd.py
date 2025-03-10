from typing import List, Type

from galois import FieldArray
from oraqle.compiler.nodes.fpd.abstract import FpNode
from oraqle.compiler.nodes.fpd.abstract import FpdNode
from oraqle.compiler.nodes.fpd.galois import FieldNorm
from oraqle.compiler.nodes.zpx_cycl_pow2.abstract import MultAutParetoFront, PolyRingPow2, ZpxNode


def poly_eval_galois_within_degree(gf: Type[FieldArray], coefficients: List[int], element: FpNode) -> FpNode:
    assert (len(coefficients) - 1) <= gf.degree

    added_coefficients, factor, inverse_factor, poly_degree = None  # TODO: Call Rust code

    # Add a poly and multiply with a factor to ensure the degree is a power of two and the poly is monic
    if added poly is nonzero
    new_coefficients = list(coefficients)
    new_coefficients.extend([0] * (len(added_coefficients) - len(new_coefficients)))
    p = gf.characteristic
    mul only if inv is not one
    new_coefficients = [(((a + b) % p) * inverse_factor) % p for a, b in zip(added_coefficients, new_coefficients)]

    # Perform the norm computation
    element_fpd = FpdNode(element, degree=poly_degree)
    alpha: FpdNode = None  # TODO: Call Rust code, also give the characteristic poly?
    res = FieldNorm(alpha - element, poly_degree)

    # Undo the added_poly and the factor
    if inverse_factor != 1:
        res = res * inverse_factor
    if added_poly nonzero:
        res += 

    return res


# TODO: This should probably be a UnivariateZpxNode
class UnivariatePolyZpx(ZpxNode):
    pass
