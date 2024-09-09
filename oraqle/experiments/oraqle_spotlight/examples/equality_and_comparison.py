from galois import GF

from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.leafs import Input

if __name__ == "__main__":
    gf = GF(31)

    x = Input("x", gf)
    y = Input("y", gf)
    z = Input("z", gf)

    comparison = x < y
    equality = y == z
    both = comparison & equality

    circuit = Circuit([both])

    circuit.to_pdf("example.pdf")
