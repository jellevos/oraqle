"""Arithmetization of an AND operation between 15 inputs."""

from galois import GF

from oraqle.compiler.boolean.bool_and import And
from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.abstract import UnoverloadedWrapper
from oraqle.compiler.nodes.leafs import Input

gf = GF(5)

xs = [Input(f"x{i}", gf) for i in range(15)]

output = And(set(UnoverloadedWrapper(x) for x in xs), gf)

circuit = Circuit(outputs=[output])
circuit.to_graph("high_level_circuit.dot")

arithmetic_circuit = circuit.arithmetize()
arithmetic_circuit.to_graph("arithmetic_circuit.dot")
