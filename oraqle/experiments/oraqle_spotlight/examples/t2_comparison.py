from galois import GF

from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.leafs import Input

p = 7
gf = GF(p)

x = Input("x", gf)
y = Input("y", gf)

comparison = 0

for a in range((p + 1) // 2, p):
    comparison += 1 - (x - y - a) ** (p - 1)

circuit = Circuit([comparison])  # type: ignore

if __name__ == "__main__":
    circuit.to_graph("t2.dot")
    circuit.to_pdf("t2.pdf")
