"""This module contains classes and functions for efficient exponentiation circuits."""
import math
from typing import Type

from galois import GF, FieldArray

from oraqle.add_chains.addition_chains_front import gen_pareto_front
from oraqle.add_chains.addition_chains_heuristic import add_chain_guaranteed
from oraqle.add_chains.solving import extract_indices
from oraqle.compiler.nodes.abstract import CostParetoFront, Node
from oraqle.compiler.nodes.binary_arithmetic import Multiplication
from oraqle.compiler.nodes.leafs import Input
from oraqle.compiler.nodes.univariate import UnivariateNode


# TODO: Think about the role of Power when there are also Products
class Power(UnivariateNode):
    """Represents an exponentiation: x ** constant."""

    @property
    def _node_shape(self) -> str:
        return "box"

    @property
    def _hash_name(self) -> str:
        return f"pow_{self._exponent}"

    @property
    def _node_label(self) -> str:
        return f"Pow: {self._exponent}"

    def __init__(self, node: Node, exponent: int, gf: Type[FieldArray]):
        """Initialize a `Power` node that exponentiates `node` with `exponent`."""
        self._exponent = exponent
        super().__init__(node, gf)

    def _operation_inner(self, input: FieldArray, gf: Type[FieldArray]) -> FieldArray:
        return input**self._exponent  # type: ignore

    def _arithmetize_inner(self, strategy: str) -> "Node":
        if strategy == "naive":
            # Square & multiply
            nodes = [self._node.arithmetize(strategy)]

            for i in range(math.ceil(math.log2(self._exponent))):
                nodes.append(nodes[i].mul(nodes[i], flatten=False))
            previous = None
            for i in range(math.ceil(math.log2(self._exponent))):
                if (self._exponent >> i) & 1:
                    if previous is None:
                        previous = nodes[i]
                    else:
                        nodes.append(nodes[i].mul(previous, flatten=False))
                        previous = nodes[-1]

            assert previous is not None
            return previous

        assert strategy == "best-effort"

        addition_chain = add_chain_guaranteed(self._exponent, self._gf.characteristic - 1, squaring_cost=1.0)

        nodes = [self._node.arithmetize(strategy).to_arithmetic()]

        for i, j in addition_chain:
            nodes.append(Multiplication(nodes[i], nodes[j], self._gf))

        return nodes[-1]

    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        # TODO: While generating the front, we can take into account the maximum cost etc. implied by the depth-aware arithmetization of the operand
        if self._gf.characteristic <= 257:
            front = gen_pareto_front(self._exponent, self._gf.characteristic, cost_of_squaring)
        else:
            front = gen_pareto_front(self._exponent, None, cost_of_squaring)

        final_front = CostParetoFront(cost_of_squaring)

        for depth1, _, node in self._node.arithmetize_depth_aware(cost_of_squaring):
            for depth2, chain in front:
                c = extract_indices(
                    chain,
                    modulus=self._gf.characteristic - 1 if self._gf.characteristic <= 257 else None,
                )

                nodes = [node]

                for i, j in c:
                    nodes.append(Multiplication(nodes[i], nodes[j], self._gf))

                final_front.add(nodes[-1], depth=depth1 + depth2)

        return final_front


def test_depth_aware_arithmetization():  # noqa: D103
    gf = GF(31)

    x = Input("x", gf)
    node = Power(x, 30, gf)
    front = node.arithmetize_depth_aware(cost_of_squaring=1.0)
    node.clear_cache(set())

    for _, _, n in front:
        assert n.evaluate({"x": gf(0)}) == 0
        n.clear_cache(set())

        for xx in range(1, 31):
            assert n.evaluate({"x": gf(xx)}) == 1
