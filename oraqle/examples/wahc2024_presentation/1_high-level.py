"""Renders a high-level comparison circuit."""
from galois import GF

from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.fp.leafs import FpInput


if __name__ == "__main__":
    gf = GF(101)

    alex = FpInput("a", gf)
    blake = FpInput("b", gf)

    output = alex < blake
    circuit = Circuit(outputs=[output])

    circuit.to_svg("high_level.svg")
