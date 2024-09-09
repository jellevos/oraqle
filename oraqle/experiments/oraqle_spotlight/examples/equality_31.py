from galois import GF

from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.leafs import Input

if __name__ == "__main__":
    gf = GF(31)

    x = Input("x", gf)
    y = Input("y", gf)

    equality = x == y

    circuit = Circuit([equality])
    arithmetic_circuits = circuit.arithmetize_depth_aware(cost_of_squaring=1.0)

    for d, _, arithmetic_circuit in arithmetic_circuits:
        arithmetic_circuit.to_pdf(f"equality_{d}.pdf")
