"""Visualization of three circuits computing an OR operation on 7 inputs."""

from galois import GF

from oraqle.compiler.arithmetic.exponentiation import Power
from oraqle.compiler.boolean.bool_neg import Neg
from oraqle.compiler.circuit import ArithmeticCircuit, Circuit
from oraqle.compiler.nodes.binary_arithmetic import Multiplication
from oraqle.compiler.nodes.leafs import Input

gf = GF(5)

x1 = Input("x1", gf)
x2 = Input("x2", gf)
x3 = Input("x3", gf)
x4 = Input("x4", gf)
x5 = Input("x5", gf)
x6 = Input("x6", gf)
x7 = Input("x7", gf)

sum1 = x1 + x2 + x3 + x4
exp1 = Power(sum1, 4, gf)

sum2 = x5 + x6 + x7 + exp1
exp2 = Power(sum2, 4, gf)

circuit = Circuit([exp2])
arithmetic_circuit = circuit.arithmetize()
arithmetic_circuit.to_graph("arithmetic_circuit1.dot")


inv1 = Neg(x1, gf)
inv2 = Neg(x2, gf)
inv3 = Neg(x3, gf)
inv4 = Neg(x4, gf)
inv5 = Neg(x5, gf)
inv6 = Neg(x6, gf)

mul1 = inv1 * inv2
invmul1 = Neg(mul1, gf)

mul2 = inv3 * inv4
invmul2 = Neg(mul2, gf)

mul3 = inv5 * inv6
invmul3 = Neg(mul3, gf)

add1 = mul1 + mul2
add2 = mul3 + add1

add3 = add2 + x7

exp = Power(add3, 4, gf)

circuit = Circuit([exp])
arithmetic_circuit = circuit.arithmetize()
arithmetic_circuit.to_graph("arithmetic_circuit2.dot")


inv1 = Neg(x1, gf).arithmetize("best-effort").to_arithmetic()
inv2 = Neg(x2, gf).arithmetize("best-effort").to_arithmetic()
inv3 = Neg(x3, gf).arithmetize("best-effort").to_arithmetic()
inv4 = Neg(x4, gf).arithmetize("best-effort").to_arithmetic()
inv5 = Neg(x5, gf).arithmetize("best-effort").to_arithmetic()
inv6 = Neg(x6, gf).arithmetize("best-effort").to_arithmetic()
inv7 = Neg(x7, gf).arithmetize("best-effort").to_arithmetic()

mul1 = Multiplication(inv1, inv2, gf)
mul2 = Multiplication(inv3, inv4, gf)
mul3 = Multiplication(inv5, inv6, gf)

mul4 = Multiplication(mul1, mul2, gf)
mul5 = Multiplication(mul3, inv7, gf)

mul6 = Multiplication(mul4, mul5, gf)

inv = Neg(mul6, gf).arithmetize("best-effort").to_arithmetic()

arithmetic_circuit = ArithmeticCircuit([inv])
arithmetic_circuit.to_graph("arithmetic_circuit3.dot")
