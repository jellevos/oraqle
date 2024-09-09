import math
import multiprocessing
import pickle
import time
from functools import partial
from typing import List, Tuple

from matplotlib import pyplot as plt
from sympy import sieve

from oraqle.add_chains.addition_chains_front import chain_depth, gen_pareto_front
from oraqle.add_chains.addition_chains_mod import chain_cost, hw


def experiment(
    t: int, squaring_cost: float
) -> Tuple[List[Tuple[int, float, List[Tuple[int, int]]]], float]:
    start = time.monotonic()
    chains = gen_pareto_front(
        t - 1,
        modulus=t - 1,
        squaring_cost=squaring_cost,
        solver="glucose42",
        encoding=1,
        thurber=True,
    )
    duration = time.monotonic() - start

    return [
        (chain_depth(chain, modulus=t - 1), chain_cost(chain, squaring_cost), chain)
        for _, chain in chains
    ], duration


def experiment2(
    t: int, squaring_cost: float
) -> Tuple[List[Tuple[int, float, List[Tuple[int, int]]]], float]:
    start = time.monotonic()
    chains = gen_pareto_front(
        t - 1,
        modulus=None,
        squaring_cost=squaring_cost,
        solver="glucose42",
        encoding=1,
        thurber=True,
    )
    duration = time.monotonic() - start

    return [
        (chain_depth(chain), chain_cost(chain, squaring_cost), chain) for _, chain in chains
    ], duration


def plot_specific_outputs(specific_outputs, specific_outputs_nomod, primes, squaring_cost: float):
    plt.figure(figsize=(9, 2.8))
    plt.grid(axis="y", zorder=-1000, alpha=0.5)

    for x, p in enumerate(primes):
        label = "Square & multiply" if p == 2 else None
        t = p - 1
        plt.scatter(
            x,
            math.ceil(math.log2(t)) * squaring_cost + hw(t) - 1,
            color="black",
            label=label,
            zorder=100,
            marker="_",
        )

    for x, outputs in enumerate(specific_outputs):
        chains, _ = outputs
        for depth, cost, _ in chains:
            plt.scatter(
                x,
                cost,
                color="black",
                zorder=100,
                s=50,
                label="Optimal circuit" if x == 0 else None,
            )
            if len(chains) > 1:
                plt.text(
                    x,
                    cost - 0.05,
                    str(depth),
                    fontsize=6,
                    ha="center",
                    va="center",
                    color="white",
                    zorder=200,
                    fontweight="bold",
                )

    plt.xticks(range(len(primes)), primes, rotation=50)
    plt.yticks(range(2 * math.ceil(math.log2(primes[-1]))))

    plt.xlabel("Modulus")
    plt.ylabel("Multiplicative cost")

    ax1 = plt.gca()
    ax2 = ax1.twinx()
    for x, outputs in enumerate(specific_outputs):
        _, duration = outputs
        ax2.bar(x, duration, color="tab:cyan", zorder=0, alpha=0.3, label="Considering modulus" if x == 0 else None)  # type: ignore
    for x, outputs in enumerate(specific_outputs_nomod):
        _, duration = outputs
        ax2.bar(x, duration, color="tab:cyan", zorder=0, alpha=1.0, label="Ignoring modulus" if x == 0 else None)  # type: ignore
    ax2.set_ylabel("Generation time [s]", color="tab:cyan", alpha=1.0)

    ax1.step(
        range(len(primes)),
        [squaring_cost * math.ceil(math.log2(p - 1)) for p in primes],
        zorder=10,
        color="black",
        where="mid",
        label="Lower bound",
        linestyle=":",
    )

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()  # type: ignore
    ax1.legend(lines + lines2, labels + labels2, loc="upper left", fontsize="small")

    plt.savefig(f"equality_first_prime_mods_{squaring_cost}.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    run_experiments = False

    if run_experiments:
        multiprocessing.set_start_method("fork")
        threads = 4
        pool = multiprocessing.Pool(threads)

        primes = list(sieve.primerange(300))[:30]  # [:50]

        for sqr_cost in [0.5, 0.75, 1.0]:
            print(f"Computing for {sqr_cost}")
            experiment_sqr_cost = partial(experiment, squaring_cost=sqr_cost)
            outs = list(pool.map(experiment_sqr_cost, primes))

            with open(f"equality_experiment_{sqr_cost}_mod.pkl", mode="wb") as file:
                pickle.dump((primes, outs), file)

        for sqr_cost in [0.5, 0.75, 1.0]:
            print(f"Computing for {sqr_cost}")
            experiment_sqr_cost = partial(experiment2, squaring_cost=sqr_cost)
            outs = list(pool.map(experiment_sqr_cost, primes))

            with open(f"equality_experiment_{sqr_cost}_nomod.pkl", mode="wb") as file:
                pickle.dump((primes, outs), file)

    # Visualize
    with open("equality_experiment_0.5_mod.pkl", "rb") as file:
        primes_05_mod, outputs_05_mod = pickle.load(file)
    with open("equality_experiment_0.75_mod.pkl", "rb") as file:
        primes_075_mod, outputs_075_mod = pickle.load(file)
    with open("equality_experiment_1.0_mod.pkl", "rb") as file:
        primes_10_mod, outputs_10_mod = pickle.load(file)

    with open("equality_experiment_0.5_nomod.pkl", "rb") as file:
        primes_05_nomod, outputs_05_nomod = pickle.load(file)
    with open("equality_experiment_0.75_nomod.pkl", "rb") as file:
        primes_075_nomod, outputs_075_nomod = pickle.load(file)
    with open("equality_experiment_1.0_nomod.pkl", "rb") as file:
        primes_10_nomod, outputs_10_nomod = pickle.load(file)

    # All the primes should match
    primes = primes_10_mod
    assert primes == primes_05_mod
    assert primes == primes_075_mod
    assert primes == primes_05_nomod
    assert primes == primes_075_nomod
    assert primes == primes_10_nomod

    # All the chains should match (not in theory, but for this visualization they should)
    assert all(
        all(x == y for x, y in zip(a[0], b[0])) for a, b in zip(outputs_05_mod, outputs_05_nomod)
    )
    assert all(
        all(x == y for x, y in zip(a[0], b[0])) for a, b in zip(outputs_075_mod, outputs_075_nomod)
    )
    assert all(
        all(x == y for x, y in zip(a[0], b[0])) for a, b in zip(outputs_10_mod, outputs_10_nomod)
    )

    plot_specific_outputs(outputs_05_mod, outputs_05_nomod, primes, squaring_cost=0.5)
    plot_specific_outputs(outputs_075_mod, outputs_075_nomod, primes, squaring_cost=0.75)
    plot_specific_outputs(outputs_10_mod, outputs_10_nomod, primes, squaring_cost=1.0)
