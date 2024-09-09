"""MIMC is an MPC-friendly cipher: https://eprint.iacr.org/2016/492."""
from math import ceil, log2
from random import randint

from galois import GF

from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes import Constant, Input, Node

gf = GF(680564733841876926926749214863536422929)


# TODO: Check parameters with the paper
def encrypt(plaintext: Node, key: int, power_n: int = 129) -> Node:
    """Returns an MIMC encryption circuit using a constant key."""
    rounds = ceil(power_n / log2(3))

    constants = [
        (
            Constant(gf(0))
            if (round == 0) or (round == (rounds - 1))
            else Constant(gf(randint(0, 2**power_n)))
        )
        for round in range(rounds)
    ]
    key_constant = Constant(gf(key))

    for round in range(rounds):
        added = plaintext + key_constant + constants[round]
        plaintext = added * added * added

    return plaintext + key_constant


if __name__ == "__main__":
    node = encrypt(Input("m", gf), 12345)

    circuit = Circuit([node]).arithmetize()
    print(circuit.multiplicative_depth())
    print(circuit.multiplicative_size())

    circuit.to_graph("mimc-129.dot")


def test_mimc_129():  # noqa: D103
    node = encrypt(Input("m", gf), 12345)

    circuit = Circuit([node]).arithmetize()

    assert circuit.multiplicative_depth() == 164
    assert circuit.multiplicative_size() == 164
