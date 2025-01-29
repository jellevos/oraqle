from galois import GF

from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.leafs import Input


if __name__ == "__main__":
    iterations = 10

    for p in [29, 43, 61, 101, 131]:
        gf = GF(p)

        x = Input("x", gf)
        y = Input("y", gf)

        circuit = Circuit([x == y])

        for d, c, arith in circuit.arithmetize_depth_aware(0.75):
            print(d, c, arith.run_using_helib(10, True, False, x=13, y=19))

        arith = circuit.arithmetize('naive')
        print('square and multiply', arith.multiplicative_depth(), arith.multiplicative_size(), arith.multiplicative_cost(0.75), arith.run_using_helib(10, True, False, x=13, y=19))
