"""Depth-aware arithmetization for an equality operation modulo 31."""

from galois import GF

from oraqle.compiler.circuit import Circuit
from oraqle.compiler.comparison.equality import Equals
from oraqle.compiler.nodes.leafs import Input

gf = GF(31)

a = Input("a", gf)
b = Input("b", gf)

output = Equals(a, b, gf)

circuit = Circuit(outputs=[output])

arithmetic_circuits = circuit.arithmetize_depth_aware(cost_of_squaring=1.0)

if __name__ == "__main__":
    circuit.to_pdf("high_level_circuit.pdf")
    for depth, size, arithmetic_circuit in arithmetic_circuits:
        arithmetic_circuit.to_pdf(f"arithmetic_circuit_d{depth}_s{size}.pdf")
