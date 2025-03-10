"""Renders two circuits, one with a balanced product tree and one with an imbalanced tree."""
from galois import GF

from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.fp.leafs import FpInput

if __name__ == "__main__":
    gf = GF(101)

    a = FpInput("a", gf)
    b = FpInput("b", gf)
    c = FpInput("c", gf)
    d = FpInput("d", gf)

    output = a * b * c * d
    circuit_good = Circuit(outputs=[output])
    circuit_good = circuit_good.arithmetize_depth_aware()  # FIXME: This should also work with arithmetize
    circuit_good[0][2].to_svg("rebalancing_good.svg")

    ab = a.mul(b, flatten=False)
    abc = ab.mul(c, flatten=False)
    abcd = abc.mul(d, flatten=False)
    circuit_bad = Circuit(outputs=[abcd])
    circuit_bad = circuit_bad.arithmetize()
    circuit_bad.to_svg("rebalancing_bad.svg")
