"""This module contains tools for memoizing addition chains, as these are expensive to compute."""
from hashlib import sha3_256
import inspect
import shelve
from typing import Set

from sympy import sieve


ADDCHAIN_CACHE_PATH = "addchain_cache"


# Adapted from: https://stackoverflow.com/questions/16463582/memoize-to-disk-python-persistent-memoization
def cache_to_disk(file_name, ignore_args: Set[str]):
    """This decorator caches the calls to this function in a file on disk, ignoring the arguments listed in `ignore_args`.
    
    Returns:
        A cached output
    """
    d = shelve.open(file_name)  # noqa: SIM115

    def decorator(func):
        signature = inspect.signature(func)
        signature_args = list(signature.parameters.keys())
        assert all(arg in signature_args for arg in ignore_args)
        
        def wrapped_func(*args, **kwargs):
            relevant_args = [a for a, sa in zip(args, signature_args) if sa not in ignore_args]
            for kwarg in signature_args[len(args):]:
                if kwarg not in ignore_args:
                    relevant_args.append(kwargs[kwarg])

            h = sha3_256()
            h.update(str(relevant_args).encode('ascii'))
            hashed_args = h.hexdigest()
            
            if hashed_args not in d:
                d[hashed_args] = func(*args, **kwargs)
            return d[hashed_args]

        return wrapped_func

    return decorator


if __name__ == "__main__":
    from oraqle.add_chains.addition_chains_front import gen_pareto_front

    # Precompute addition chains for x^(p-1) mod p for the first 30 primes p
    primes = list(sieve.primerange(300))[:30]
    for sqr_cost in [0.5, 0.75, 1.0]:
        print(f"Computing for {sqr_cost}")
        
        for p in primes:
            gen_pareto_front(
                p - 1,
                modulus=p - 1,
                squaring_cost=sqr_cost,
                solver="glucose42",
                encoding=1,
                thurber=True,
            )
