"""Generates code for the comparison circuit."""
from galois import GF

from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.leafs import Input

if __name__ == "__main__":
    gf = GF(101)

    alex = Input("a", gf)
    blake = Input("b", gf)

    output = alex < blake
    circuit = Circuit(outputs=[output])

    front = circuit.arithmetize_depth_aware()

    for _, _, arithmetic_circuit in front:
        program = arithmetic_circuit.generate_code("example.cpp")
