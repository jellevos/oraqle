from galois import GF

from oraqle.compiler.boolean.bool_and import all_
from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.leafs import Input

if __name__ == "__main__":
    gf = GF(17)

    xs = (Input(f"x{i + 1}", gf) for i in range(16))

    conjunction = all_(*xs)

    circuit = Circuit([conjunction])
    arithmetic_circuit = circuit.arithmetize()

    arithmetic_circuit.to_pdf("conjunction.pdf")
