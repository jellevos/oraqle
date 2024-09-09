from typing import Tuple

from galois import GF

from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.abstract import Node
from oraqle.compiler.nodes.arbitrary_arithmetic import sum_
from oraqle.compiler.nodes.leafs import Input


def generate_nodes() -> Tuple[Node, Node]:
    gf = GF(31)

    x = Input("x", gf)
    y = Input("y", gf)
    z1 = Input("z1", gf)
    z2 = Input("z2", gf)
    z3 = Input("z3", gf)
    z4 = Input("z4", gf)

    comparison = x < y
    sum = sum_(z1, z2, z3, z4)
    cse1 = comparison & sum

    comparison = y > x
    sum = sum_(z3, z2, z4) + z1
    cse2 = sum & comparison

    return cse1, cse2


def test_cse_equivalence():
    cse1, cse2 = generate_nodes()
    assert cse1.is_equivalent(cse2)


if __name__ == "__main__":
    cse1, cse2 = generate_nodes()

    cse1 = Circuit([cse1])
    cse2 = Circuit([cse2])

    cse1.to_pdf("cse1.pdf")
    cse2.to_pdf("cse2.pdf")
