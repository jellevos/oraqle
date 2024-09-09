"""This module contains classes for checking if an element is in the upper half of the finite field."""
import math

from galois import GF, FieldArray

from oraqle.add_chains.addition_chains_front import gen_pareto_front
from oraqle.add_chains.addition_chains_heuristic import add_chain_guaranteed
from oraqle.add_chains.solving import extract_indices
from oraqle.compiler.nodes.abstract import CostParetoFront, Node
from oraqle.compiler.nodes.binary_arithmetic import Addition, Multiplication
from oraqle.compiler.nodes.leafs import Input
from oraqle.compiler.nodes.unary_arithmetic import ConstantMultiplication
from oraqle.compiler.nodes.univariate import UnivariateNode
from oraqle.compiler.polynomials.univariate import UnivariatePoly, _eval_poly


class InUpperHalf(UnivariateNode):
    """Returns 1 when the input is contained in the upper half of the field, which are considered the negative numbers.

    Specifically, it returns 0 in the range [0, (p - 1) / 2] and 1 in the range ((p - 1) / 2, p - 1].
    """

    @property
    def _node_shape(self) -> str:
        return "box"

    @property
    def _hash_name(self) -> str:
        return "in_upper_half"

    @property
    def _node_label(self) -> str:
        return "> (p-1)/2"

    def _operation_inner(self, input: FieldArray) -> FieldArray:
        p = self._gf.characteristic
        if 0 <= int(input) <= p // 2:
            return self._gf(0)

        return self._gf(1)

    def _arithmetize_inner(self, strategy: str) -> Node:
        coefficients = []

        # From: Faster homomorphic comparison operations for BGV and BFV, Ilia Iliashenko & Vincent Zucca, 2021
        p = self._gf.characteristic
        for i in range(p - 1):
            if i % 2 == 0:
                # Ignore every even power, we take care of this by squaring the input node.
                continue

            coefficient = self._gf(0)
            for a in range(1, p // 2 + 1):
                coefficient += self._gf(a) ** (p - 1 - i)
            coefficients.append(coefficient)

        # We do not add the final coefficient, which will be computed later, so we do not do coefficients.append(gf((p + 1) // 2))

        input_node = self._node.arithmetize(strategy).to_arithmetic()
        input_node_squared = input_node * input_node
        arithmetization, precomputed_powers = UnivariatePoly(
            input_node_squared, coefficients, self._gf
        ).arithmetize_custom(strategy)

        # Since we skip the first coefficient, we manually multiply the output by the input node.
        result = Multiplication(input_node, arithmetization, self._gf)

        # Compute the final coefficient using an exponentiation
        precomputed_values = tuple(
            (
                (2 * exp) % (p - 1),
                power_node.multiplicative_depth() - input_node.multiplicative_depth(),
            )
            for exp, power_node in precomputed_powers.items()
        )
        
        addition_chain = add_chain_guaranteed(p - 1, p - 1, squaring_cost=1.0, precomputed_values=precomputed_values)

        nodes = [input_node]
        nodes.extend(power_node for _, power_node in precomputed_powers.items())

        for i, j in addition_chain:
            nodes.append(Multiplication(nodes[i], nodes[j], self._gf))

        final_term = ConstantMultiplication(nodes[-1], self._gf((p + 1) // 2))

        return (Addition(result, final_term, self._gf)).arithmetize(strategy)

    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        # TODO: Handle p = 2 and p = 3 separately

        # TODO: Reduce code duplication
        final_front = CostParetoFront(cost_of_squaring)

        for node_depth, _, node in self._node.arithmetize_depth_aware(cost_of_squaring):
            coefficients = []

            # From: Faster homomorphic comparison operations for BGV and BFV, Ilia Iliashenko & Vincent Zucca, 2021
            p = self._gf.characteristic
            for i in range(p - 1):
                if i % 2 == 0:
                    # Ignore every even power, we take care of this by squaring the input node.
                    continue

                coefficient = self._gf(0)
                for a in range(1, p // 2 + 1):
                    coefficient += self._gf(a) ** (p - 1 - i)
                coefficients.append(coefficient)

            # We do not add the final coefficient, which will be computed later, so we do not do coefficients.append(gf((p + 1) // 2))

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
                    for exp, power_node in precomputed_powers[depth].items()
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
                    nodes.extend(power_node for _, power_node in precomputed_powers[depth].items())

                    for i, j in c:
                        nodes.append(Multiplication(nodes[i], nodes[j], self._gf))

                    final_power_front.add(nodes[-1], depth=node_depth + depth2)

                for _, _, final_power in final_power_front:
                    final_term = ConstantMultiplication(final_power, self._gf((p + 1) // 2))
                    final_front.add(Addition(result, final_term, self._gf))

        assert not final_front.is_empty()
        return final_front


class IliashenkoZuccaInUpperHalf(UnivariateNode):
    """Returns 1 when the input is contained in the upper half of the field, which are considered the negative numbers.

    Specifically, it returns 0 in the range [0, (p - 1) / 2] and 1 in the range ((p - 1) / 2, p - 1].
    """

    @property
    def _node_shape(self) -> str:
        return "box"

    @property
    def _hash_name(self) -> str:
        return "in_upper_half_iz21"

    @property
    def _node_label(self) -> str:
        return "> (p-1)/2 [IZ21]"

    def _operation_inner(self, input: FieldArray) -> FieldArray:
        p = self._gf.characteristic
        if 0 <= int(input) <= p // 2:
            return self._gf(0)

        return self._gf(1)

    def _arithmetize_inner(self, strategy: str) -> Node:
        coefficients = []

        # TODO: This is copied from above
        # From: Faster homomorphic comparison operations for BGV and BFV, Ilia Iliashenko & Vincent Zucca, 2021
        p = self._gf.characteristic
        for i in range(p - 1):
            if i % 2 == 0:
                # Ignore every even power, we take care of this by squaring the input node.
                continue

            coefficient = self._gf(0)
            for a in range(1, p // 2 + 1):
                coefficient += self._gf(a) ** (p - 1 - i)
            coefficients.append(coefficient)

        # We do not add the final coefficient, which will be computed later, so we do not do coefficients.append(gf((p + 1) // 2))

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
            for exp, power_node in precomputed_powers.items()
        )
        
        addition_chain = add_chain_guaranteed(p - 1, p - 1, squaring_cost=1.0, precomputed_values=precomputed_values)

        nodes = [input_node]
        nodes.extend(power_node for _, power_node in precomputed_powers.items())

        for i, j in addition_chain:
            nodes.append(Multiplication(nodes[i], nodes[j], self._gf))
        final_monomial = nodes[-1]

        final_term = ConstantMultiplication(final_monomial, self._gf((p + 1) // 2))

        return (Addition(result, final_term, self._gf)).arithmetize(strategy)

    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        raise NotImplementedError()
        # TODO: Handle p = 2 and p = 3 separately


# TODO: Make a univariate node class with an easy way to test if evaluation corresponds to evaluating the arithmetic
def test_evaluate_mod7():  # noqa: D103
    gf = GF(7)

    x = Input("x", gf)
    node = InUpperHalf(x, gf)

    for i in range(3):
        assert node.evaluate({"x": gf(i)}) == gf(0)
        node.clear_cache(set())
    for i in range(4, 7):
        assert node.evaluate({"x": gf(i)}) == gf(1)
        node.clear_cache(set())


def test_evaluate_arithmetized_mod7():  # noqa: D103
    gf = GF(7)

    x = Input("x", gf)
    node = InUpperHalf(x, gf).arithmetize("best-effort")
    node.clear_cache(set())

    for i in range(3):
        assert node.evaluate({"x": gf(i)}) == gf(0)
        node.clear_cache(set())
    for i in range(4, 7):
        assert node.evaluate({"x": gf(i)}) == gf(1)
        node.clear_cache(set())
