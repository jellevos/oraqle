"""Show the first step of arithmetization of a comparison circuit."""
from galois import GF

from oraqle.compiler.boolean.bool_neg import Neg
from oraqle.compiler.circuit import Circuit
from oraqle.compiler.comparison.comparison import SemiStrictComparison
from oraqle.compiler.nodes.leafs import Constant, Input

if __name__ == "__main__":
    gf = GF(101)

    alex = Input("a", gf)
    blake = Input("b", gf)

    output = alex < blake


    p = output._gf.characteristic

    if output._less_than:
        left = output._left
        right = output._right
    else:
        left = output._right
        right = output._left

    left = left.arithmetize("best-effort")
    right = right.arithmetize("best-effort")

    left_is_small = SemiStrictComparison(
        left, Constant(output._gf(p // 2)), less_than=True, gf=output._gf
    )
    right_is_small = SemiStrictComparison(
        right, Constant(output._gf(p // 2)), less_than=True, gf=output._gf
    )

    # Test whether left and right are in the same range
    same_range = (left_is_small & right_is_small) + (
        Neg(left_is_small, output._gf) & Neg(right_is_small, output._gf)
    )

    # Performs left < right on the reduced inputs, note that if both are in the upper half the difference is still small enough for a semi-comparison
    comparison = SemiStrictComparison(left, right, less_than=True, gf=output._gf)
    result = same_range * comparison

    # Performs left < right when one if small and the other is large
    right_is_larger = left_is_small & Neg(right_is_small, output._gf)
    result += right_is_larger


    circuit = Circuit(outputs=[result])

    circuit.to_svg("arith_step1.svg")
