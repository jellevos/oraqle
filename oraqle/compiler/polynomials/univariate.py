"""Evaluation of univariate polynomials."""

import math
from typing import Callable, Dict, List, Optional, Tuple, Type

from galois import GF, FieldArray

from oraqle.add_chains.addition_chains_heuristic import add_chain_guaranteed
from oraqle.compiler.arithmetic.subtraction import Subtraction
from oraqle.compiler.func2poly import interpolate_polynomial
from oraqle.compiler.nodes.abstract import ArithmeticNode, CostParetoFront, Node
from oraqle.compiler.nodes.binary_arithmetic import Multiplication
from oraqle.compiler.nodes.leafs import Constant, Input
from oraqle.compiler.nodes.unary_arithmetic import ConstantMultiplication
from oraqle.compiler.nodes.univariate import UnivariateNode
from oraqle.config import PS_METHOD_FACTOR_K


def _format_polynomial(coefficients: List[FieldArray]) -> str:
    degree = len(coefficients) - 1
    if degree == 0:
        return str(coefficients[0])

    terms = []
    for i, coef in enumerate(coefficients):
        if coef == 0:
            # Skip zero coefficients
            continue

        term = str(coef) if i == 0 or coef > 1 else ""

        if i > 0:
            term += "x"

        if i > 1:
            term += f"^{i}"

        if term != "":
            terms.append(term)

    polynomial = " + ".join(terms)
    return polynomial


class UnivariatePoly(UnivariateNode):
    """Evaluation of a univariate polynomial."""

    @property
    def _node_shape(self) -> str:
        return "box"

    @property
    def _hash_name(self) -> str:
        return "univariate_poly"

    @property
    def _node_label(self) -> str:
        return _format_polynomial(self._coefficients)

    def __init__(
        self,
        node: Node,
        coefficients: List[FieldArray],
        gf: Type[FieldArray],
    ):
        """Initialize a univariate polynomial with the given coefficients from least to highest order."""
        self._coefficients = coefficients
        # TODO: We can reduce this polynomial if its degree is too high
        super().__init__(node, gf)

        self._custom_arithmetize_cache = None

    @classmethod
    def from_function(
        cls, node: Node, gf: Type[FieldArray], function: Callable[[int], int]
    ) -> "UnivariatePoly":
        """Interpolate a univariate polynomial for the given function.
        
        Returns:
        -------
        A UnivariatePoly whose coefficients compute the `function` on all inputs.

        """
        coefficients = [
            gf(int(coeff) % gf.characteristic)
            for coeff in reversed(
                interpolate_polynomial(function, gf.characteristic, ["x"]).as_list()
            )
        ]
        return cls(node, coefficients, gf)

    def _operation_inner(self, input: FieldArray) -> FieldArray:
        coefficient_iter = iter(self._coefficients)
        result = next(coefficient_iter).copy()

        x_pow = input.copy()
        for coefficient in coefficient_iter:
            result += coefficient * x_pow
            x_pow *= input

        return result  # type: ignore

    def _arithmetize_inner(self, strategy: str) -> Node:
        return self.arithmetize_custom(strategy)[0]

    def arithmetize_custom(self, strategy: str) -> Tuple[ArithmeticNode, Dict[int, ArithmeticNode]]:
        """Compute an arithmetization along with a dictionary of precomputed powers.

        Returns:
        -------
        An arithmetization and a dictionary of previously computed powers.

        """
        if len(self._coefficients) == 0:
            return Constant(self._gf(0)), {}

        if len(self._coefficients) == 1:
            return Constant(self._coefficients[0]), {}

        x = self._node.arithmetize(strategy).to_arithmetic()

        best_arithmetization: Optional[Node] = None
        best_arithmetization_powers = None

        lowest_multiplicative_size = 1_000_000_000  # TODO: Not elegant
        optimal_k = math.sqrt(2 * len(self._coefficients))
        bound = min(int(math.ceil(PS_METHOD_FACTOR_K * optimal_k)), len(self._coefficients))
        for k in range(1, bound):
            (
                arithmetization,
                precomputed_powers,
            ) = _eval_poly(x, self._coefficients, k, self._gf, 1.0)

            arithmetization = arithmetization.to_arithmetic()
            # TODO: It would be best to perform CSE during the circuit creation
            assert isinstance(arithmetization, ArithmeticNode)

            if arithmetization.multiplicative_size() <= lowest_multiplicative_size:
                lowest_multiplicative_size = arithmetization.multiplicative_size()
                best_arithmetization = arithmetization
                best_arithmetization_powers = precomputed_powers

            # TODO: Also perform the alternative poly evaluation

        # TODO: This check is probably unnecessary
        assert best_arithmetization is not None
        assert best_arithmetization_powers is not None

        return (
            best_arithmetization.arithmetize(strategy),
            best_arithmetization_powers,
        )

    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        return self.arithmetize_depth_aware_custom(cost_of_squaring)[0]

    def arithmetize_depth_aware_custom(
        self, cost_of_squaring: float
    ) -> Tuple[CostParetoFront, Dict[int, Dict[int, ArithmeticNode]]]:
        """Compute a depth-aware arithmetization as well as a dictionary indexed by the depth of the nodes in the front. The dictionary stores precomputed powers.

        Returns:
        -------
        A CostParetoFront with the depth-aware arithmetization and a dictionary indexed by the depth of the nodes in the front, returning a dictionary with previously computed powers.

        """
        # TODO: Perhaps this should be cached
        if len(self._coefficients) == 0:
            return CostParetoFront.from_leaf(Constant(self._gf(0)), cost_of_squaring), {0: {}}

        if len(self._coefficients) == 1:
            return CostParetoFront.from_leaf(Constant(self._coefficients[0]), cost_of_squaring), {
                0: {}
            }

        front = CostParetoFront(cost_of_squaring)
        all_precomputed_powers = {}

        for _, _, x in self._node.arithmetize_depth_aware(cost_of_squaring):
            optimal_k = math.sqrt(2 * len(self._coefficients))
            bound = min(int(math.ceil(PS_METHOD_FACTOR_K * optimal_k)), len(self._coefficients))
            for k in range(1, bound):
                (
                    arithmetization,
                    precomputed_powers,
                ) = _eval_poly(x, self._coefficients, k, self._gf, cost_of_squaring)

                arithmetization = arithmetization.to_arithmetic()
                assert isinstance(arithmetization, ArithmeticNode)

                added = front.add(arithmetization)
                if added:
                    all_precomputed_powers[arithmetization.multiplicative_depth()] = (
                        precomputed_powers
                    )

            for k in range(1, len(self._coefficients)):
                (
                    arithmetization,
                    precomputed_powers,
                ) = _eval_poly_divide_conquer(x, self._coefficients, k, self._gf, cost_of_squaring)

                arithmetization = arithmetization.to_arithmetic()
                assert isinstance(arithmetization, ArithmeticNode)

                added = front.add(arithmetization)
                if added:
                    all_precomputed_powers[arithmetization.multiplicative_depth()] = (
                        precomputed_powers
                    )

            for k in range(1, len(self._coefficients)):
                (
                    arithmetization,
                    precomputed_powers,
                ) = _eval_poly_alternative(x, self._coefficients, k, self._gf)

                arithmetization = arithmetization.to_arithmetic()
                assert isinstance(arithmetization, ArithmeticNode)

                added = front.add(arithmetization)
                if added:
                    all_precomputed_powers[arithmetization.multiplicative_depth()] = (
                        precomputed_powers
                    )

        precomputed_powers = {depth: all_precomputed_powers[depth] for depth, _, _ in front}
        return front, precomputed_powers


def _monic_euclidean_division(
    a: List[FieldArray], b: List[FieldArray], gf
) -> Tuple[List[FieldArray], List[FieldArray]]:
    q = [gf(0) for _ in range(len(a))]
    r = [el.copy() for el in a]
    d = len(b) - 1
    c = b[-1].copy()
    assert c == 1
    while (len(r) - 1) >= d:
        if r[-1] == 0:
            r.pop()
            continue

        s_monomial = len(r) - 1 - d
        f = r[-1]
        q[s_monomial] += f

        for i in range(d + 1):
            r[s_monomial + i] -= f * b[i]
        r.pop()

    while len(q) > 0 and q[-1] == 0:
        q.pop()

    return q, r


def _eval_poly_using_precomputed_ks(
    coefficients: List[FieldArray], precomputed_ks: List[ArithmeticNode], gf
) -> ArithmeticNode:
    if len(coefficients) == 0:
        return Constant(gf(0))

    # TODO: What if the constant is 0? Do we want to rely on no-op removal later or do it here already?
    output = Constant(coefficients[0])

    for i in range(1, len(coefficients)):
        if coefficients[i] == 0:
            continue

        if coefficients[i] == 1:
            output += precomputed_ks[i - 1]
            continue

        output += (
            Constant(coefficients[i]).mul(precomputed_ks[i - 1], flatten=False)
        )  # FIXME: Consider just using *

    return output.arithmetize("best-effort").to_arithmetic()


def _eval_monic_poly_specific(
    coefficients: List[FieldArray],
    precomputed_ks: List[ArithmeticNode],
    precomputed_pow2s: List[ArithmeticNode],
    gf,
    p: int,
) -> ArithmeticNode:
    if all(c == 0 for c in coefficients):
        return Constant(gf(0))

    degree = len(coefficients) - 1

    # Base case, this is free after precomputation
    if degree <= len(precomputed_ks):
        return _eval_poly_using_precomputed_ks(coefficients, precomputed_ks, gf)

    assert degree % len(precomputed_ks) == 0
    assert ((degree // len(precomputed_ks)) + 1) % 2 == 0

    k = len(precomputed_ks)
    assert p == (((degree // k) + 1) // 2)

    r = coefficients[: (k * p - 1) + 1]
    q = coefficients[(k * p - 1) + 1 :]

    assert (len(q) - 1) == k * (p - 1)

    r[k * (p - 1)] = r[k * (p - 1)].copy() - gf(1)
    c, s = _monic_euclidean_division(r, q, gf)
    assert len(c) - 1 <= (len(precomputed_ks) - 1)

    monomial = precomputed_pow2s[int(math.log2(p))]

    c_output = _eval_poly_using_precomputed_ks(c, precomputed_ks, gf)

    left = monomial.add(c_output, flatten=False)
    right = _eval_monic_poly_specific(q, precomputed_ks, precomputed_pow2s, gf, p // 2)

    s.append(gf(1))  # This adds the monomial
    assert (len(s) - 1) == k * (p - 1)
    remainder = _eval_monic_poly_specific(s, precomputed_ks, precomputed_pow2s, gf, p // 2)

    final_product = left.mul(right, flatten=False)
    return (
        final_product.add(remainder, flatten=False).arithmetize("best-effort").to_arithmetic()
    )  # TODO: Strategy


def _precompute_ks(x: ArithmeticNode, k: int) -> List[ArithmeticNode]:
    # TODO: We can use an addition sequence for this to reduce the multiplicative cost
    ks = [x]
    for _ in range(math.ceil(math.log2(k))):
        last = ks[-1]
        new_ks = []
        for pre in ks:
            new_ks.append(Multiplication(pre, last, pre._gf))
        ks.extend(new_ks)

    return ks[:k]


def _compute_extended_monomial(
    x: ArithmeticNode,
    precomputed_powers: Dict[int, ArithmeticNode],
    target: int,
    gf: Type[FieldArray],
    squaring_cost: float,
) -> ArithmeticNode:
    if target == 0:
        return Constant(gf(1))

    # TODO: Use squaring_cost
    p = gf.characteristic
    precomputed_values = tuple(
        (
            exp % (p - 1),
            power_node.multiplicative_depth() - x.multiplicative_depth(),
        )
        for exp, power_node in precomputed_powers.items()
    )
    # TODO: This is copied from Power, but in the future we can probably remove this if we have augmented circuits
    addition_chain = add_chain_guaranteed(target, modulus=p - 1, squaring_cost=squaring_cost, precomputed_values=precomputed_values)

    nodes = [x]
    nodes.extend(power_node for _, power_node in precomputed_powers.items())

    for i, j in addition_chain:
        nodes.append(Multiplication(nodes[i], nodes[j], gf))

    return nodes[-1]


def _eval_poly(
    x: ArithmeticNode,
    coefficients: List[FieldArray],
    k: int,
    gf: Type[FieldArray],
    squaring_cost: float,
) -> Tuple[ArithmeticNode, Dict[int, ArithmeticNode]]:
    # Paterson & Stockmeyer's algorithm
    degree = len(coefficients) - 1
    precomputed_ks = _precompute_ks(x, k)
    precomputed_powers = {
        i % (gf.characteristic - 1): node for i, node in zip(range(1, k + 1), precomputed_ks)
    }

    # Find the largest p such that k(2^p-1) >= degree
    p = 0
    while True:
        p += 1
        if (2**p - 1) * k >= degree:
            break

    new_degree = (2**p - 1) * k
    precomputed_pow2s = [precomputed_ks[-1]]
    for j in range(p - 1):  # TODO: Check if p - 1 is enough
        precomputed_pow2s.append(
            Multiplication(precomputed_pow2s[-1], precomputed_pow2s[-1], precomputed_pow2s[-1]._gf)
        )
        precomputed_powers[(k * (2 ** (j + 1))) % (gf.characteristic - 1)] = precomputed_pow2s[-1]

    # Pad to the next degree k * (2^p - 1) monic polynomial
    new_coefficients = [gf(0) for _ in range(new_degree + 1)]
    for j, c in enumerate(coefficients):
        new_coefficients[j] = c.copy()

    extended = new_coefficients[-1] == 0
    factor = gf(1)
    if int(new_coefficients[-1]) > 1:
        # The polynomial is not monic
        inverse = coefficients[-1] ** -1
        new_coefficients = [inverse * c for c in coefficients]
        factor = coefficients[-1]

    new_coefficients[-1] = gf(1)

    monomial_index = new_degree % (gf.characteristic - 1)
    if monomial_index == 0:
        monomial_index = gf.characteristic - 1
    if extended and monomial_index <= degree:
        # In some cases we can eliminate the added monomial by changing the coefficients
        new_coefficients[monomial_index] -= gf(1)
        extended = False

    evaluation = _eval_monic_poly_specific(
        new_coefficients, precomputed_ks, precomputed_pow2s, gf, 2**p // 2
    )

    if extended:
        monomial = _compute_extended_monomial(
            x, precomputed_powers, new_degree % (gf.characteristic - 1), gf, squaring_cost
        )
        precomputed_powers[new_degree % (gf.characteristic - 1)] = monomial
        evaluation = (
            Subtraction(evaluation, monomial, gf).arithmetize("best-effort").to_arithmetic()
        )  # TODO: We should not have to choose a strategy here

    if int(factor) > 1:
        # Make up for the missing factor
        evaluation = ConstantMultiplication(evaluation, factor)

    return evaluation, precomputed_powers


def _eval_poly_alternative(
    x: ArithmeticNode, coefficients: List[FieldArray], k: int, gf: Type[FieldArray]
) -> Tuple[Node, Dict[int, ArithmeticNode]]:
    # Baby-step giant-step algorithm
    assert len(coefficients) > 0

    i = len(coefficients) - 1
    while coefficients[i] == 0:
        i -= 1
    coefficients = [coefficients[j].copy() for j in range(i + 1)]  # Copies and trims the coefficients

    # Precompute x, x^2, ..., x^k
    precomputed_ks = _precompute_ks(x, k)
    precomputed_powers = {
        i % (gf.characteristic - 1): node for i, node in zip(range(1, k + 1), precomputed_ks)
    }

    # Process the first chunk
    chunk = coefficients[-(k + 1) :]
    aggregator = _eval_poly_using_precomputed_ks(chunk, precomputed_ks, gf)
    coefficients = coefficients[: -(k + 1)]

    # Go through the coefficients, chunk by chunk
    while len(coefficients) >= k:
        chunk = coefficients[-k:]
        aggregator = aggregator * precomputed_ks[-1] + _eval_poly_using_precomputed_ks(
            chunk, precomputed_ks, gf
        )
        coefficients = coefficients[:-k]

    # If there is a small chunk remaining
    if len(coefficients) > 0:
        aggregator = aggregator * precomputed_ks[
            len(coefficients) - 1
        ] + _eval_poly_using_precomputed_ks(coefficients, precomputed_ks, gf)

    return aggregator, precomputed_powers


def _eval_poly_divide_conquer_specific(
    coefficients: List[FieldArray],
    precomputed_ks: List[ArithmeticNode],
    precomputed_pow2s: List[ArithmeticNode],
    gf,
    p: int,
) -> ArithmeticNode:
    if all(c == 0 for c in coefficients):
        return Constant(gf(0))

    degree = len(coefficients) - 1

    # Base case, this is free after precomputation
    if degree <= len(precomputed_ks):
        return _eval_poly_using_precomputed_ks(coefficients, precomputed_ks, gf)

    assert degree / 2 <= (len(precomputed_ks) * p)

    subdegree = p * len(precomputed_ks)
    r = coefficients[:subdegree]
    q = coefficients[subdegree:]

    r_eval = _eval_poly_divide_conquer_specific(r, precomputed_ks, precomputed_pow2s, gf, p // 2)
    q_eval = _eval_poly_divide_conquer_specific(q, precomputed_ks, precomputed_pow2s, gf, p // 2)

    final_product = q_eval.mul(precomputed_pow2s[int(math.log2(p))], flatten=False)
    return (
        final_product.add(r_eval, flatten=False).arithmetize("best-effort").to_arithmetic()
    )  # TODO: Strategy


def _eval_poly_divide_conquer(
    x: ArithmeticNode,
    coefficients: List[FieldArray],
    k: int,
    gf: Type[FieldArray],
    _squaring_cost: float,
) -> Tuple[ArithmeticNode, Dict[int, ArithmeticNode]]:
    # Divide-and-conquer algorithm
    # TODO: Reduce code duplication with poly_eval
    degree = len(coefficients) - 1
    precomputed_ks = _precompute_ks(x, k)
    precomputed_powers = {
        i % (gf.characteristic - 1): node for i, node in zip(range(1, k + 1), precomputed_ks)
    }

    # Find the largest p such that k * 2^p >= degree
    p = 0
    while True:
        p += 1
        if 2**p * k >= degree:
            break

    precomputed_pow2s = [precomputed_ks[-1]]
    for j in range(p - 1):  # TODO: Check if p - 1 is enough
        precomputed_pow2s.append(
            Multiplication(precomputed_pow2s[-1], precomputed_pow2s[-1], precomputed_pow2s[-1]._gf)
        )
        precomputed_powers[(k * (2 ** (j + 1))) % (gf.characteristic - 1)] = precomputed_pow2s[-1]

    evaluation = _eval_poly_divide_conquer_specific(
        coefficients, precomputed_ks, precomputed_pow2s, gf, 2 ** (p - 1)
    )

    return evaluation, precomputed_powers


def _eval_coefficients(x: FieldArray, coefficients: List[FieldArray]) -> FieldArray:
    x_pow = x.copy()
    result = coefficients[0].copy()

    for coeff in coefficients[1:]:
        result += x_pow * coeff
        x_pow *= x

    return result


def test_ps_method():  # noqa: D103
    gf = GF(31)
    coefficients = [gf(i) for i in range(31)]

    x = Input("x", gf)

    for k in range(1, len(coefficients)):
        (
            arithmetization,
            _,
        ) = _eval_poly(x, coefficients, k, gf, squaring_cost=1.0)
        arithmetization.clear_cache(set())

        for xx in range(31):
            assert arithmetization.evaluate({"x": gf(xx)}) == _eval_coefficients(gf(xx), coefficients)
            arithmetization.clear_cache(set())
    
    assert all(coefficients[i] == i for i in range(31))

    
def test_divide_conquer_method():  # noqa: D103
    gf = GF(31)
    coefficients = [gf(i) for i in range(31)]

    x = Input("x", gf)

    for k in range(1, len(coefficients)):
        (
            arithmetization,
            _,
        ) = _eval_poly_divide_conquer(x, coefficients, k, gf, _squaring_cost=1.0)
        arithmetization.clear_cache(set())

        for xx in range(31):
            assert arithmetization.evaluate({"x": gf(xx)}) == _eval_coefficients(gf(xx), coefficients)
            arithmetization.clear_cache(set())
        
    assert all(coefficients[i] == i for i in range(31))


def test_babystep_giantstep_method():  # noqa: D103
    gf = GF(31)
    coefficients = [gf(i) for i in range(31)]

    x = Input("x", gf)

    for k in range(1, len(coefficients)):
        (
            arithmetization,
            _,
        ) = _eval_poly_alternative(x, coefficients, k, gf)
        arithmetization.clear_cache(set())

        for xx in range(31):
            assert arithmetization.evaluate({"x": gf(xx)}) == _eval_coefficients(gf(xx), coefficients)
            arithmetization.clear_cache(set())

    assert all(coefficients[i] == i for i in range(31))
