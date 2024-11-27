"""Renders two circuits, one with a balanced product tree and one with an imbalanced tree."""
from galois import GF

from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.leafs import Input

if __name__ == "__main__":
    gf = GF(101)

    a = Input("a", gf)
    b = Input("b", gf)
    c = Input("c", gf)
    d = Input("d", gf)

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
