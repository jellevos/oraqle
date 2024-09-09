"""This module implements a high-level AES encryption circuit for a constant key."""
from typing import List

from aeskeyschedule import key_schedule
from galois import GF

from oraqle.compiler.arithmetic.exponentiation import Power
from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes import Constant
from oraqle.compiler.nodes.abstract import Node
from oraqle.compiler.nodes.leafs import Input

gf = GF(2**8)


def encrypt(plaintext: List[Node], key: bytes) -> List[Node]:
    """Returns an AES encryption circuit for a constant `key`."""
    mix = [Constant(gf(2)), Constant(gf(3)), Constant(gf(1)), Constant(gf(1))]

    round_keys = [[Constant(gf(byte)) for byte in round_key] for round_key in key_schedule(key)]

    def additions(nodes: List[Node]) -> Node:
        node_iter = iter(nodes)
        out = next(node_iter) + next(node_iter)
        for node in node_iter:
            out += node
        return out

    def sbox(node: Node, method="minchain") -> Node:
        if method == "hardcoded":
            x2 = node.mul(node, flatten=False)
            x3 = node.mul(x2, flatten=False)
            x6 = x3.mul(x3, flatten=False)
            x12 = x6.mul(x6, flatten=False)
            x15 = x12.mul(x3, flatten=False)
            x30 = x15.mul(x15, flatten=False)
            x60 = x30.mul(x30, flatten=False)
            x63 = x60.mul(x3, flatten=False)
            x126 = x63.mul(x63, flatten=False)
            x127 = node.mul(x126, flatten=False)
            x254 = x127.mul(x127, flatten=False)
            return x254
        elif method == "minchain":
            return Power(node, 254, gf)
        else:
            raise Exception(f"Invalid method: {method}.")

    # AddRoundKey
    b = [round_key + plaintext_byte for round_key, plaintext_byte in zip(round_keys[0], plaintext)]

    for round in range(9):
        # SubBytes (modular inverse)
        b = [sbox(b[j], method="hardcoded") for j in range(16)]

        # ShiftRows
        b[1], b[5], b[9], b[13] = b[5], b[9], b[13], b[1]
        b[2], b[6], b[10], b[14] = b[10], b[14], b[2], b[6]
        b[3], b[7], b[11], b[15] = b[15], b[3], b[7], b[11]

        # MixColumns
        b = [additions([mix[(j + i) % 4] * b[j // 4 + i] for i in range(4)]) for j in range(16)]

        # AddRoundKey
        b = [round_key + b[j] for j, round_key in zip(range(16), round_keys[round + 1])]
        b: List[Node]

    return b


if __name__ == "__main__":
    # TODO: Consider if we want to support degree > 1
    circuit = Circuit(
        encrypt([Input(f"{i}", gf) for i in range(16)], b"abcdabcdabcdabcd")
    ).arithmetize()
    print(circuit)
    print(circuit.multiplicative_depth())
    print(circuit.multiplicative_size())
    circuit.eliminate_subexpressions()
    print(circuit.multiplicative_depth())
    print(circuit.multiplicative_size())

    # TODO: Test if it corresponds to a plaintext implementation of AES


def test_aes_128():  # noqa: D103
    # Only checks if no errors occur
    Circuit(encrypt([Input(f"{i}", gf) for i in range(16)], b"abcdabcdabcdabcd")).arithmetize()
