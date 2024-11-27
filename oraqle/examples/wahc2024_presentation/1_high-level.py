"""Renders a high-level comparison circuit."""
from galois import GF

from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.leafs import Input


if __name__ == "__main__":
    gf = GF(101)

    alex = Input("a", gf)
    blake = Input("b", gf)

    output = alex < blake
    circuit = Circuit(outputs=[output])

    circuit.to_svg("high_level.svg")
