"""Arithmetizes a comparison modulo 11 with a constant."""

from galois import GF

from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.leafs import Constant, Input

gf = GF(11)

a = Input("a", gf)
b = Constant(gf(3))  # Input("b")

output = a < b

circuit = Circuit(outputs=[output])
circuit.to_graph("high_level_circuit.dot")

arithmetic_circuit = circuit.arithmetize()
arithmetic_circuit.to_graph("arithmetic_circuit.dot")
