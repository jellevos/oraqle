"""Creates graphs for the arithmetization of a small polynomial evaluation."""

from galois import GF

from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.leafs import Input
from oraqle.compiler.polynomials.univariate import UnivariatePoly

gf = GF(11)

x = Input("x", gf)

output = UnivariatePoly(x, [gf(1), gf(2), gf(3), gf(4), gf(5), gf(6), gf(1)], gf)

circuit = Circuit(outputs=[output])
circuit.to_graph("high_level_circuit.dot")

arithmetic_circuit = circuit.arithmetize()
arithmetic_circuit.to_graph("arithmetic_circuit.dot")
