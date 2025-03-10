from galois import GF

from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.fp.leafs import FpInput

if __name__ == "__main__":
    gf = GF(31)

    x = FpInput("x", gf)
    y = FpInput("y", gf)
    z = FpInput("z", gf)

    comparison = x < y
    equality = y == z
    both = comparison & equality

    circuit = Circuit([both])

    circuit.to_pdf("example.pdf")
