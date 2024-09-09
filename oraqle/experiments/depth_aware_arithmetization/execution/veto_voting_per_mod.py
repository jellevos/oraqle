from typing import List

from galois import GF
from matplotlib import pyplot as plt
from sympy import sieve

from oraqle.compiler.boolean.bool_and import _minimum_cost
from oraqle.compiler.boolean.bool_or import Or
from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.abstract import CostParetoFront, UnoverloadedWrapper
from oraqle.compiler.nodes.leafs import Input
from oraqle.experiments.oraqle_spotlight.experiments.veto_voting_minimal_cost import (
    exponentiation_results,
)


def generate_all_fronts():
    results = {}

    for p in [7, 11, 13, 17]:
        fronts = []

        print(f"------ p = {p} ------")
        for k in range(2, 51):
            gf = GF(p)
            xs = [Input(f"x{i}", gf) for i in range(k)]

            circuit = Circuit([Or(set(UnoverloadedWrapper(x) for x in xs), gf)])
            front = circuit.arithmetize_depth_aware(cost_of_squaring=1.0)

            print(f"{k}.", end=" ")
            for f in front:
                print(f[0], f[1], end="   ")

            print()
            fronts.append(front)

        results[p] = fronts

    return results


def plot_fronts(fronts: List[CostParetoFront], color, label, **kwargs):
    plt.scatter([], [], color=color, label=label, **kwargs)
    for k, front in zip(range(2, 51), fronts):
        for depth, cost, _ in front:
            kwargs["marker"] = (depth, 2, 0)
            kwargs["s"] = 16
            kwargs["linewidth"] = 0.5
            plt.scatter(k, cost, color=color, **kwargs)


if __name__ == "__main__":
    fronts_by_p = generate_all_fronts()
    max_k = 50

    plt.figure(figsize=(4, 4))

    plt.plot(
        range(2, max_k + 1),
        [k - 1 for k in range(2, max_k + 1)],
        color="gray",
        linestyle="solid",
        label="Naive",
        linewidth=0.7,
    )

    plot_fronts(fronts_by_p[7], "tab:purple", "Modulus p = 7", zorder=100)
    plot_fronts(fronts_by_p[13], "tab:green", "Modulus p = 13", zorder=100)

    best_costs = [100000000.0] * (max_k + 1)
    best_ps = [None] * (max_k + 1)
    # This is for sqr = 0.75 mul
    primes = list(sieve.primerange(300))[1:50]
    for p in primes:
        for k in range(2, max_k + 1):
            cost = _minimum_cost(k, exponentiation_results[p][0][0][1], p)
            if cost < best_costs[k - 2]:
                best_costs[k - 2] = cost
                best_ps[k - 2] = p

    plt.step(
        range(2, max_k + 1),
        best_costs[:-2],
        zorder=10,
        color="gray",
        where="mid",
        label="Lowest for any p",
        linestyle="solid",
        linewidth=0.7,
    )

    plt.legend()

    plt.xlabel("Number of operands")
    plt.ylabel("Multiplicative size")

    plt.savefig("veto_voting.pdf", bbox_inches="tight")
    plt.show()
