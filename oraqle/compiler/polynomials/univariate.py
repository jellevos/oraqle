"""Evaluation of univariate polynomials."""

from importlib.resources import files
import math
import shelve
from typing import Callable, Dict, List, Optional, Tuple, Type

from galois import GF, FieldArray

import oraqle
from oraqle.add_chains.addition_chains_front import chain_depth, gen_pareto_front
from oraqle.add_chains.addition_chains_heuristic import add_chain_guaranteed
from oraqle.add_chains.addition_chains_mod import chain_cost
from oraqle.add_chains.solving import extract_indices
from oraqle.compiler.arithmetic.subtraction import Subtraction
from oraqle.compiler.func2poly import interpolate_polynomial
from oraqle.compiler.nodes.abstract import ArithmeticNode, CostParetoFront, Node
from oraqle.compiler.nodes.binary_arithmetic import Multiplication
from oraqle.compiler.nodes.leafs import Constant, Input
from oraqle.compiler.nodes.unary_arithmetic import ConstantMultiplication
from oraqle.compiler.nodes.univariate import UnivariateNode
from oraqle.config import PS_METHOD_FACTOR_K
from numba import njit


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


def _expand_front(
        poly_eval_construction: Callable[[ArithmeticNode, List[FieldArray], int, Type[FieldArray], float], Tuple[Node, Dict[int, ArithmeticNode]]],
        lower_bounds: Callable[[ArithmeticNode, List[FieldArray], int, Type[FieldArray], float], Tuple[int, float]],
        input: ArithmeticNode,
        coefficients: List[FieldArray],
        ks: range,
        gf: Type[FieldArray],
        front: CostParetoFront,
        all_precomputed_powers: Dict[int, Dict[int, ArithmeticNode]],
        all_constructions: Dict[int, Tuple[str, int]],
        name: str,
        cost_of_squaring: float):
    # Generate an initial front of interesting values of k by computing lower bounds
    pre_front = CostParetoFront(cost_of_squaring)
    bounds = {}
    for k in ks:
        print(k, ks)
        lb_depth, lb_cost = lower_bounds(input, coefficients, k, gf, cost_of_squaring)
        #est_depth, est_cost = _estimate_ps(input, coefficients, k, gf, cost_of_squaring)
        #print(est_depth, "==", lb_depth, est_cost, "==", lb_cost)
        bounds[k] = (lb_depth, lb_cost)
        # TODO: This is very hacky (storing k instead of a node)
        pre_front.add(k, depth=lb_depth, cost=lb_cost)  # type: ignore

    for lb_depth, remainder in pre_front._nodes_by_depth.items():
        lb_cost, k = remainder  # type: ignore
        k: int
        if not front.would_improve_front(lb_depth, lb_cost):
            continue
        
        (
            arithmetization,
            precomputed_powers,
        ) = poly_eval_construction(input, coefficients, k, gf, cost_of_squaring)

        arithmetization = arithmetization.to_arithmetic()
        assert isinstance(arithmetization, ArithmeticNode)
        # TODO: Consdier removing these checks later
        assert lb_depth <= arithmetization.multiplicative_depth()
        assert lb_cost <= arithmetization.multiplicative_cost(cost_of_squaring)

        # TODO: Handle this
        added = front.add(arithmetization)
        if added:
            d = arithmetization.multiplicative_depth()
            all_precomputed_powers[d] = (
                precomputed_powers
            )
            all_constructions[d] = (name, k)

    for k in ks:
        lb_depth, lb_cost = bounds[k]
        if not front.would_improve_front(lb_depth, lb_cost):
            continue
        
        (
            arithmetization,
            precomputed_powers,
        ) = poly_eval_construction(input, coefficients, k, gf, cost_of_squaring)

        arithmetization = arithmetization.to_arithmetic()
        assert isinstance(arithmetization, ArithmeticNode)
        assert lb_depth <= arithmetization.multiplicative_depth()
        assert lb_cost <= arithmetization.multiplicative_cost(cost_of_squaring)

        # TODO: Handle this
        added = front.add(arithmetization)
        if added:
            d = arithmetization.multiplicative_depth()
            all_precomputed_powers[arithmetization.multiplicative_depth()] = (
                precomputed_powers
            )
            all_constructions[d] = (name, k)


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
        bound = min(math.ceil(PS_METHOD_FACTOR_K * optimal_k), len(self._coefficients))
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
        oraqle_path = files(oraqle)
        database_path = oraqle_path.joinpath("poly_eval_cache")
        db = shelve.open(str(database_path))  # noqa: SIM115

        coeff_modulus_hash = str(hash((tuple(int(coeff) for coeff in self._coefficients), self._gf.characteristic)))
        if coeff_modulus_hash in db:
            print("From cache!")
            all_constructions = db[coeff_modulus_hash]
            db.close()
            front = CostParetoFront(cost_of_squaring)
            all_precomputed_powers = {}

            for _, _, x in self._node.arithmetize_depth_aware(cost_of_squaring):
                for (name, k) in all_constructions:
                    name: str
                    k: int
                    if name == "ps":
                        arithmetized, precomputed = _eval_poly(x, self._coefficients, k, self._gf, cost_of_squaring)
                    elif name == "dc":
                        arithmetized, precomputed = _eval_poly_divide_conquer(x, self._coefficients, k, self._gf, cost_of_squaring)
                    elif name == "bg":
                        arithmetized, precomputed = _eval_poly_alternative(x, self._coefficients, k, self._gf, cost_of_squaring)
                    else:
                        raise Exception("Invalid name in poly db")

                    front.add(arithmetized)
                    all_precomputed_powers[arithmetized.multiplicative_depth()] = precomputed
            
            precomputed_powers = {depth: all_precomputed_powers[depth] for depth, _, _ in front}
            return front, precomputed_powers


        # TODO: Perhaps this should be cached (we can hash the coefficients along with the plaintext modulus and save which techniques and which ks led to the front)
        if len(self._coefficients) == 0:
            return CostParetoFront.from_leaf(Constant(self._gf(0)), cost_of_squaring), {0: {}}

        if len(self._coefficients) == 1:
            return CostParetoFront.from_leaf(Constant(self._coefficients[0]), cost_of_squaring), {
                0: {}
            }

        front = CostParetoFront(cost_of_squaring)
        all_precomputed_powers = {}
        all_constructions = {}

        for _, _, x in self._node.arithmetize_depth_aware(cost_of_squaring):
            optimal_k = math.sqrt(2 * len(self._coefficients))
            bound = min(math.ceil(PS_METHOD_FACTOR_K * optimal_k), len(self._coefficients))
            _expand_front(_eval_poly, _estimate_ps, x, self._coefficients, range(1, bound), self._gf, front, all_precomputed_powers, all_constructions, 'ps', cost_of_squaring)

            optimal_k = math.sqrt(len(self._coefficients))  # FIXME: Use the exact optimal k (this is not a great approximation)
            bound = min(math.ceil(PS_METHOD_FACTOR_K * optimal_k), len(self._coefficients))
            _expand_front(_eval_poly_divide_conquer, _lower_bounds_divide_conquer, x, self._coefficients, range(1, bound), self._gf, front, all_precomputed_powers, all_constructions, 'dc', cost_of_squaring)

            optimal_k = math.sqrt(len(self._coefficients))
            bound = min(math.ceil(PS_METHOD_FACTOR_K * optimal_k), len(self._coefficients))
            _expand_front(_eval_poly_alternative, _lower_bounds_alternative, x, self._coefficients, range(1, bound), self._gf, front, all_precomputed_powers, all_constructions, 'bg', cost_of_squaring)

        db[coeff_modulus_hash] = list(all_constructions[depth] for depth, _, _ in front)
        db.close()

        precomputed_powers = {depth: all_precomputed_powers[depth] for depth, _, _ in front}
        return front, precomputed_powers


@njit
def _monic_euclidean_division_njit(
    a: List[int], b: List[int], p: int
) -> Tuple[List[int], List[int]]:
    q = [0 for _ in range(len(a))]
    r = [el for el in a]
    d = len(b) - 1
    #c = b[-1]
    #assert c == 1
    while (len(r) - 1) >= d:
        if r[-1] == 0:
            r.pop()
            continue

        s_monomial = len(r) - 1 - d
        f = r[-1]
        q[s_monomial] += f
        q[s_monomial] %= p

        for i in range(d + 1):
            r[s_monomial + i] -= f * b[i]
            r[s_monomial + i] %= p
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
    coefficients: List[int],
    precomputed_ks: List[ArithmeticNode],
    precomputed_pow2s: List[ArithmeticNode],
    gf: Type[FieldArray],
    p: int,
) -> ArithmeticNode:
    if all(c == 0 for c in coefficients):
        return Constant(gf(0))

    degree = len(coefficients) - 1

    # Base case, this is free after precomputation
    if degree <= len(precomputed_ks):
        return _eval_poly_using_precomputed_ks([gf(el) for el in coefficients], precomputed_ks, gf)

    assert degree % len(precomputed_ks) == 0
    assert ((degree // len(precomputed_ks)) + 1) % 2 == 0

    k = len(precomputed_ks)
    assert p == (((degree // k) + 1) // 2)

    r = coefficients[: (k * p - 1) + 1]
    q = coefficients[(k * p - 1) + 1 :]

    assert (len(q) - 1) == k * (p - 1)

    r[k * (p - 1)] = r[k * (p - 1)] - 1
    r[k * (p - 1)] %= gf.characteristic
    c, s = _monic_euclidean_division_njit(r, q, gf.characteristic)
    assert len(c) - 1 <= (len(precomputed_ks) - 1)

    monomial = precomputed_pow2s[int(math.log2(p))]

    c_output = _eval_poly_using_precomputed_ks([gf(el) for el in c], precomputed_ks, gf)

    left = monomial.add(c_output, flatten=False)
    right = _eval_monic_poly_specific(q, precomputed_ks, precomputed_pow2s, gf, p // 2)

    s.append(1)  # This adds the monomial
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
    max_depth: int) -> ArithmeticNode:
    if target == 0:
        return Constant(gf(1))

    p = gf.characteristic
    precomputed_values = tuple(
        (
            exp % (p - 1),
            power_node.multiplicative_depth() - x.multiplicative_depth(),
        )
        for exp, power_node in precomputed_powers.items()
    )
    # TODO: This is copied from Power, but in the future we can probably remove this if we have augmented circuits
    addition_chain = add_chain_guaranteed(target, modulus=p - 1, squaring_cost=squaring_cost, precomputed_values=precomputed_values, max_depth=max_depth)
    # print('Prec', precomputed_values)
    # print('Chain', addition_chain)
    # front = gen_pareto_front(target, modulus=p - 1, squaring_cost=squaring_cost, precomputed_values=precomputed_values)
    # print('Front', front)
    # exit(0)

    nodes = [x]
    nodes.extend(power_node for _, power_node in precomputed_powers.items())

    for i, j in addition_chain:
        nodes.append(Multiplication(nodes[i], nodes[j], gf))

    return nodes[-1]


def _estimate_ps(x: ArithmeticNode, coefficients: List[FieldArray], k: int, gf: Type[FieldArray], cost_of_squaring: float) -> Tuple[int, float]:
    # TODO: Skip trailing 0s
    degree = len(coefficients) - 1

    ## Generate the new coefficients
    # Find the largest p such that k(2^p-1) >= degree
    qq = 0
    while True:
        qq += 1
        if (2**qq - 1) * k >= degree:
            break

    # Estimate depth and cost
    depth = math.ceil(math.log2(k)) + qq  # x.multiplicative_depth() + 
    cost = 2**(qq - 1) - 1 + (k - 1)

    # FIXME:
    # # Handle extension
    new_degree = (2**qq - 1) * k
    extended = False
    if new_degree > degree:
        extended = True

    monomial_index = new_degree % (gf.characteristic - 1)
    if monomial_index == 0:
        monomial_index = gf.characteristic - 1
    if extended and monomial_index <= degree:
        # In some cases we can eliminate the added monomial by changing the coefficients
        extended = False

    if extended:
        precomputed_ks = _precompute_ks(x, k)
        precomputed_powers = {
            i % (gf.characteristic - 1): node for i, node in zip(range(1, k + 1), precomputed_ks)
        }
        precomputed_pow2s = [precomputed_ks[-1]]
        for j in range(qq - 1):  # TODO: Check if p - 1 is enough
            precomputed_pow2s.append(
                Multiplication(precomputed_pow2s[-1], precomputed_pow2s[-1], precomputed_pow2s[-1]._gf)
            )
            precomputed_powers[(k * (2 ** (j + 1))) % (gf.characteristic - 1)] = precomputed_pow2s[-1]

        monomial = _compute_extended_monomial(
            x, precomputed_powers, new_degree % (gf.characteristic - 1), gf, cost_of_squaring, max_depth=depth
        )

        depth = max(monomial.multiplicative_depth(), depth + x.multiplicative_depth())
        # TODO: cost += monomial.multiplicative_cost(cost_of_squaring)

    return depth, cost

def _lower_bounds_ps(x: ArithmeticNode, coefficients: List[FieldArray], k: int, gf: Type[FieldArray], cost_of_squaring: float) -> Tuple[int, float]:
    # TODO: Skip trailing 0s
    degree = len(coefficients) - 1
    precomputed_ks = _precompute_ks(x, k)
    precomputed_powers = {
        i % (gf.characteristic - 1): node for i, node in zip(range(1, k + 1), precomputed_ks)
    }

    ## Generate the new coefficients
    # Find the largest p such that k(2^p-1) >= degree
    qq = 0
    while True:
        qq += 1
        if (2**qq - 1) * k >= degree:
            break

    new_degree = (2**qq - 1) * k
    precomputed_pow2s = [precomputed_ks[-1]]
    for j in range(qq - 1):  # TODO: Check if p - 1 is enough
        precomputed_pow2s.append(
            Multiplication(precomputed_pow2s[-1], precomputed_pow2s[-1], precomputed_pow2s[-1]._gf)
        )
        precomputed_powers[(k * (2 ** (j + 1))) % (gf.characteristic - 1)] = precomputed_pow2s[-1]

    # Pad to the next degree k * (2^p - 1) monic polynomial
    new_coefficients = [gf(0) for _ in range(new_degree + 1)]
    for j, c in enumerate(coefficients):
        new_coefficients[j] = c.copy()  # TODO: It would be more efficient if we didn't need copies

    extended = new_coefficients[-1] == 0
    if int(new_coefficients[-1]) > 1:
        # The polynomial is not monic
        inverse = coefficients[-1] ** -1
        new_coefficients = [inverse * c for c in new_coefficients]

    new_coefficients[-1] = gf(1)

    monomial_index = new_degree % (gf.characteristic - 1)
    if monomial_index == 0:
        monomial_index = gf.characteristic - 1
    if extended and monomial_index <= degree:
        # In some cases we can eliminate the added monomial by changing the coefficients
        new_coefficients[monomial_index] -= gf(1)
        extended = False

    ## Find an addition chain for the extension
    cost = x.multiplicative_cost(cost_of_squaring) + (k - 1) + (qq - 1) * cost_of_squaring
    p = gf.characteristic
    if extended:
        precomputed_values = tuple(
            (
                exp % (p - 1),
                power_node.multiplicative_depth() - x.multiplicative_depth(),
            )
            for exp, power_node in precomputed_powers.items()
        )
        # TODO: This is copied from Power, but in the future we can probably remove this if we have augmented circuits
        monomial_index = new_degree % (gf.characteristic - 1)
        if monomial_index == 0:
            monomial_index = gf.characteristic - 1
        addition_chain = add_chain_guaranteed(monomial_index, modulus=p - 1, squaring_cost=cost_of_squaring, precomputed_values=precomputed_values)
        # FIXME: We are currently ignoring this!!!
        #cost += chain_cost(addition_chain, cost_of_squaring)

    ## Recurse
    never_used_precomps = {i for i in range(k)}
    polys = [(2**qq // 2, [int(el) for el in new_coefficients])]
    while len(polys) > 0:
        pp, coeffs = polys.pop()
        if len(coeffs) - 1 <= k:
            if len(never_used_precomps) > 0:
                removing = []
                for never_used_precomp in never_used_precomps:
                    if never_used_precomp >= len(coeffs):
                        continue
                    if coeffs[never_used_precomp] != 0:
                        removing.append(never_used_precomp)
                for never_used_precomp in removing:
                    never_used_precomps.remove(never_used_precomp)
            continue

        r = coeffs[: (k * pp - 1) + 1]
        q = coeffs[(k * pp - 1) + 1 :]

        r[k * (pp - 1)] -= 1
        r[k * (pp - 1)] %= p
        #c, s = _monic_euclidean_division(r, q, gf)
        c, s = _monic_euclidean_division_njit(r, q, p)

        s.append(1)

        # Right (q)
        right_const = all(coeff == 0 for coeff in q)
        if not right_const:
            polys.append((pp // 2, q))
        right_const |= len(q) == 1
        
        # Remainder (s)
        remainder_const = all(coeff == 0 for coeff in s)
        if not remainder_const:
            polys.append((pp // 2, s))

        cost += not right_const

    depth = x.multiplicative_depth() + math.ceil(math.log2(k)) + qq
    cost -= len(never_used_precomps)

    # FIXME: We are currently ignoring this!!
    # if extended:
    #     nodes = [x]
    #     nodes.extend(power_node for _, power_node in precomputed_powers.items())

    #     for i, j in addition_chain:
    #         nodes.append(Multiplication(nodes[i], nodes[j], gf))

    #     depth = max(depth, nodes[-1].multiplicative_depth())
    
    return depth, cost


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
        new_coefficients = [inverse * c for c in new_coefficients]
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
        [int(el) for el in new_coefficients], precomputed_ks, precomputed_pow2s, gf, 2**p // 2
    )

    if extended:
        depth = evaluation.multiplicative_depth()
        monomial = _compute_extended_monomial(
            x, precomputed_powers, new_degree % (gf.characteristic - 1), gf, squaring_cost, max_depth=depth
        )
        precomputed_powers[new_degree % (gf.characteristic - 1)] = monomial
        evaluation = (
            Subtraction(evaluation, monomial, gf).arithmetize("best-effort").to_arithmetic()
        )  # TODO: We should not have to choose a strategy here

    if int(factor) > 1:
        # Make up for the missing factor
        evaluation = ConstantMultiplication(evaluation, factor)

    return evaluation, precomputed_powers


def poly_degree(coefficients: List[FieldArray]) -> int:
    for i in reversed(range(len(coefficients))):
        if coefficients[i] != 0:
            return i + 1
    return 0


# TODO: Maybe make a separate function to estimate instead of computing a lower bound
def _lower_bounds_alternative(x: ArithmeticNode, coefficients: List[FieldArray], k: int, gf: Type[FieldArray], cost_of_squaring: float) -> Tuple[int, float]:
    degree = poly_degree(coefficients)

    iterations = 0
    if degree > k + 1:
        iterations += math.ceil((degree - (k + 1)) / k)
    log_k = math.log2(k)

    depth = x.multiplicative_depth() + iterations + math.ceil(log_k)
    cost = x.multiplicative_cost(cost_of_squaring) + iterations  # + math.floor(log_k) * (cost_of_squaring - 1) + math.ceil(log_k)
    # TODO: Instead of this very loose cost bound, check if there are 0s in all strides

    # Check if there are any precomputed values that we could skip because the multiplied coefficients are always 0
    offset = (degree - 1) % k
    count = (degree - 1) // k
    selected = [any(coefficients[offset + i + j * k] != 0 for j in range(count)) for i in range(k - 1)]
    for i in range(offset):
        selected[i] |= coefficients[i] != 0
    cost += sum(selected)

    return depth, cost


def _eval_poly_alternative(
    x: ArithmeticNode, coefficients: List[FieldArray], k: int, gf: Type[FieldArray], cost_of_squaring: float,
) -> Tuple[ArithmeticNode, Dict[int, ArithmeticNode]]:
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

    return aggregator.arithmetize("best-effort").to_arithmetic(), precomputed_powers


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
    # TODO: Skip trailing 0s
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


def _lower_bounds_divide_conquer(x: ArithmeticNode, coefficients: List[FieldArray], k: int, gf: Type[FieldArray],  cost_of_squaring: float) -> Tuple[int, float]:
    # TODO: Skip trailing 0s
    degree = len(coefficients) - 1

    # Find the largest p such that k * 2^p >= degree
    p = 0
    while True:
        p += 1
        if 2**p * k >= degree:
            break

    never_used_precomps = {i for i in range(k)}
    cost = x.multiplicative_cost(cost_of_squaring) + (k - 1) + (p - 1) * cost_of_squaring
    ranges = [(2**(p - 1), 0, len(coefficients))]
    while len(ranges) > 0:
        pp, lo, hi = ranges.pop()
        if (hi - lo) - 1 <= k:
            if len(never_used_precomps) > 0:
                removing = []
                for never_used_precomp in never_used_precomps:
                    if lo + never_used_precomp >= hi or hi >= len(coefficients):
                        continue
                    if coefficients[lo + never_used_precomp] != 0:
                        removing.append(never_used_precomp)
                for never_used_precomp in removing:
                    never_used_precomps.remove(never_used_precomp)
            continue

        half = pp * k
        ppd2 = pp // 2

        # Left side
        left = coefficients[lo:lo + half]
        left_const = all(coeff == 0 for coeff in left)
        if not left_const:
            ranges.append((ppd2, lo, lo + half))
        
        # Right side
        right = coefficients[lo + half:hi]
        right_const = all(coeff == 0 for coeff in right)
        if not right_const:
            ranges.append((ppd2, lo + half, hi))
        right_const |= len(left) == 1

        cost += not right_const

    depth = x.multiplicative_depth() + math.ceil(math.log2(k)) + p - 1  # TODO: Double check the minus 1
    cost -= len(never_used_precomps)

    return depth, cost


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
        ) = _eval_poly_alternative(x, coefficients, k, gf, 1.0)
        arithmetization.clear_cache(set())

        for xx in range(31):
            assert arithmetization.evaluate({"x": gf(xx)}) == _eval_coefficients(gf(xx), coefficients)
            arithmetization.clear_cache(set())

    assert all(coefficients[i] == i for i in range(31))
