"""Show the last step of arithmetization of a comparison circuit."""
from galois import GF

from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.leafs import Input

if __name__ == "__main__":
    gf = GF(101)

    alex = Input("a", gf)
    blake = Input("b", gf)

    output = alex < blake

    front = output.arithmetize_depth_aware(cost_of_squaring=1.0)
    print(front)

    _, tup = front._nodes_by_depth.popitem()
    _, node = tup
    circuit = Circuit(outputs=[node])

    circuit.to_svg("arith_step2.svg")
