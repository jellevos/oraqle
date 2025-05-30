import math
from typing import List, Type

from galois import GF, FieldArray
from oraqle.add_chains.addition_chains_front import gen_pareto_front
from oraqle.add_chains.addition_chains_heuristic import add_chain_guaranteed
from oraqle.add_chains.solving import extract_indices
from oraqle.compiler.comparison.in_upper_half import mod_pow
from oraqle.compiler.nodes.abstract import CostParetoFront, Node
from oraqle.compiler.nodes.binary_arithmetic import Addition, Multiplication
from oraqle.compiler.nodes.leafs import Input
from oraqle.compiler.nodes.unary_arithmetic import ConstantMultiplication
from oraqle.compiler.nodes.univariate import UnivariateNode
from numba import njit

from oraqle.compiler.polynomials.univariate import UnivariatePoly, _eval_poly


@njit
def compute_coeffs(p: int, m: int) -> List[int]:
    n = p - 1
    # 1) build initial v[a] = a^(p-2) mod p  for a=1…n
    #    this is a^(−1) mod p, i.e. the inverse of a
    v = [mod_pow(a, p - 2, p) for a in range(1, n + 1)]

    # 2) build step w[a] = a^(p-3) mod p = a^(−2) mod p
    #    so multiplying by w[a] divides by a^2 mod p
    w = [mod_pow(a, p - 3, p) for a in range(1, n + 1)]

    # Also multiply the v's by a mod m
    v = [(el * (a % m)) % p for el, a in zip(v, range(1, n + 1))]

    coefficients = []
    # We want i = 1, 3, 5, ..., p - 2
    # The exponent p-2i starts at p - 2 and decreases by 2 each time
    for _ in range(1, p-1, 2):
        coefficient = sum(v) % p
        coefficients.append((p - coefficient) % p)
        for j in range(n):
            v[j] = (v[j] * w[j]) % p

    return coefficients


class Remainder(UnivariateNode):

    @property
    def _node_shape(self) -> str:
        return "box"

    @property
    def _hash_name(self) -> str:
        return f"modulo_{self._m}"

    @property
    def _node_label(self) -> str:
        return f"% {self._m}"

    def __init__(self, node: Node, modulus: int):
        self._m = modulus
        super().__init__(node, node._gf)

    def _arithmetize_inner(self, strategy: str) -> Node:
        raise NotImplementedError("TODO!")
    
    def _operation_inner(self, input: FieldArray) -> FieldArray:
        return self._gf(int(input) % self._m)

    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        # TODO: Handle p = 2 and p = 3 separately

        # TODO: Reduce code duplication
        final_front = CostParetoFront(cost_of_squaring)
        assert self._gf.degree == 1

        # From: Integer Functions Suitable for Homomorphic Encryption over Finite Fields, Ilia Iliashenko, Christophe Negre, and Vincent Zucca, 2021
        p = self._gf.characteristic
        coefficients = [self._gf(coeff) for coeff in compute_coeffs(p, self._m)]

        print("test")
        for node_depth, _, node in self._node.arithmetize_depth_aware(cost_of_squaring):
            # We do not add the final coefficient, which will be computed later

            input_node_squared = Multiplication(node, node, self._gf)
            arithmetizations, precomputed_powers = UnivariatePoly(
                input_node_squared, coefficients, self._gf
            ).arithmetize_depth_aware_custom(cost_of_squaring)

            assert not arithmetizations.is_empty()

            for depth, _, poly_arith in arithmetizations:
                # Since we skip the first coefficient, we manually multiply the output by the input node.
                result = Multiplication(node, poly_arith, self._gf)

                # Compute the final coefficient using an exponentiation
                precomputed_values = tuple(
                    ((2 * exp) % (p - 1), power_node.multiplicative_depth() - node_depth)
                    for exp, power_node in precomputed_powers[depth].items() if ((2 * exp) % (p - 1)) != 0
                )
                # TODO: This is copied from Power, but in the future we can probably remove this if we have augmented circuits
                if p <= 200:
                    front = gen_pareto_front(
                        p - 1,
                        self._gf.characteristic - 1,
                        cost_of_squaring,
                        precomputed_values=precomputed_values,
                    )
                else:
                    front = gen_pareto_front(
                        p - 1, None, cost_of_squaring, precomputed_values=precomputed_values
                    )

                final_power_front = CostParetoFront(cost_of_squaring)

                for depth2, chain in front:
                    c = extract_indices(
                        chain,
                        precomputed_values=list(k for k, _ in precomputed_values),
                        modulus=p - 1,
                    )

                    nodes = [node]
                    nodes.extend(power_node for exp, power_node in precomputed_powers[depth].items() if ((2 * exp) % (p - 1)) != 0)

                    for i, j in c:
                        nodes.append(Multiplication(nodes[i], nodes[j], self._gf))

                    final_power_front.add(nodes[-1], depth=node_depth + depth2)

                highest_coeff = (p + 1) * (self._m - 1) // 2
                highest_coeff %= p
                for _, _, final_power in final_power_front:
                    final_term = ConstantMultiplication(final_power, self._gf(highest_coeff))
                    final_front.add(Addition(result, final_term, self._gf))

        assert not final_front.is_empty()
        return final_front
    

class IliashenkoZuccaRemainder(UnivariateNode):

    @property
    def _node_shape(self) -> str:
        return "box"

    @property
    def _hash_name(self) -> str:
        return f"modulo_{self._m}_iz21"

    @property
    def _node_label(self) -> str:
        return f"% {self._m} [IZ21]"

    def __init__(self, node: Node, modulus: int):
        self._m = modulus
        super().__init__(node, node._gf)
    
    def _operation_inner(self, input: FieldArray) -> FieldArray:
        return self._gf(int(input) % self._m)

    def _arithmetize_inner(self, strategy: str) -> Node:
        coefficients = []

        # TODO: This is copied from above
        # From: Faster homomorphic comparison operations for BGV and BFV, Ilia Iliashenko & Vincent Zucca, 2021
        p = self._gf.characteristic
        coefficients = [self._gf(coeff) for coeff in compute_coeffs(p, self._m)]

        # We do not add the final coefficient, which will be computed later

        input_node = self._node.arithmetize(strategy).to_arithmetic()
        input_node_squared = Multiplication(input_node, input_node, self._gf)

        # We decide ahead of time which k to use
        k = round(math.sqrt((p - 3) / 2))
        arithmetization, precomputed_powers = _eval_poly(
            input_node_squared, coefficients, k, self._gf, squaring_cost=1.0
        )

        # Since we skip the first coefficient, we manually multiply the output by the input node.
        result = Multiplication(input_node, arithmetization, self._gf)

        # Compute the final coefficient using an exponentiation
        precomputed_values = tuple(
            (
                (2 * exp) % (p - 1),
                power_node.multiplicative_depth() - input_node.multiplicative_depth(),
            )
            for exp, power_node in precomputed_powers.items() if ((2 * exp) % (p - 1)) != 0
        )
        
        addition_chain = add_chain_guaranteed(p - 1, p - 1, squaring_cost=1.0, precomputed_values=precomputed_values)

        nodes = [input_node]
        nodes.extend(power_node for exp, power_node in precomputed_powers.items() if ((2 * exp) % (p - 1)) != 0)

        for i, j in addition_chain:
            nodes.append(Multiplication(nodes[i], nodes[j], self._gf))
        final_monomial = nodes[-1]

        highest_coeff = (p + 1) * (self._m - 1) // 2
        highest_coeff %= p
        final_term = ConstantMultiplication(final_monomial, self._gf(highest_coeff))

        return (Addition(result, final_term, self._gf)).arithmetize(strategy)

    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        raise NotImplementedError()


def test_p11_m4_depth_aware():
    p = 11
    m = 4
    gf = GF(p)

    x = Input("x", gf)
    res = Remainder(x, m).arithmetize_depth_aware(1.0)

    for _, _, node in res:
        node.clear_cache(set())

        for i in range(p):
            assert node.evaluate({"x": gf(i)}) == gf(i % m)
            node.clear_cache(set())
