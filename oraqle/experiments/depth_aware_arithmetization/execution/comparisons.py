from galois import GF

from oraqle.compiler.circuit import Circuit
from oraqle.compiler.comparison.comparison import (
    IliashenkoZuccaSemiLessThan,
    SemiStrictComparison,
    T2SemiLessThan,
)
from oraqle.compiler.nodes.leafs import Input

if __name__ == "__main__":
    for p in [29, 43, 61, 101, 131]:
        gf = GF(p)

        x = Input("x", gf)
        y = Input("y", gf)

        print(f"-------- p = {p}: ---------")
        our_circuit = Circuit([SemiStrictComparison(x, y, less_than=True, gf=gf)])
        our_front = our_circuit.arithmetize_depth_aware()
        print("Our circuits:", our_front)

        our_front[0][2].to_graph(f"comp_{p}_ours.dot")

        t2_circuit = Circuit([T2SemiLessThan(x, y, gf)])
        t2_arithmetization = t2_circuit.arithmetize()
        print(
            "T2 circuit:",
            t2_arithmetization.multiplicative_depth(),
            t2_arithmetization.multiplicative_size(),
        )
        t2_arithmetization.eliminate_subexpressions()
        print(
            "T2 circuit CSE:",
            t2_arithmetization.multiplicative_depth(),
            t2_arithmetization.multiplicative_size(),
        )

        iz21_circuit = Circuit([IliashenkoZuccaSemiLessThan(x, y, gf)])
        iz21_arithmetization = iz21_circuit.arithmetize()
        iz21_arithmetization.to_graph(f"comp_{p}_iz21.dot")
        print(
            "IZ21 circuits:",
            iz21_arithmetization.multiplicative_depth(),
            iz21_arithmetization.multiplicative_size(),
        )
        iz21_arithmetization.eliminate_subexpressions()
        iz21_arithmetization.to_graph(f"comp_{p}_iz21_cse.dot")
        print(
            "IZ21 circuit CSE:",
            iz21_arithmetization.multiplicative_depth(),
            iz21_arithmetization.multiplicative_size(),
        )
