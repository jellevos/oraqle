"""This module contains global configuration options.

!!! warning
    This is almost certainly going to be removed in the future.
We do not want oraqle to have a global configuration, but this is currently an intentional evil to prevent large refactors in the initial versions.
"""
from typing import Annotated, Optional


Seconds = Annotated[float, "seconds"]
MAXSAT_TIMEOUT: Optional[Seconds] = None
"""Time-out for individual calls to the MaxSAT solver.

!!! danger
    This causes non-deterministic behavior!
    
!!! bug
    There is currently a chance to get `AttributeError`s, which is a bug caused by pysat trying to delete an oracle that does not exist.
    There is no current workaround for this."""


PS_METHOD_FACTOR_K: float = 2.0
"""Approximation factor for the PS-method, higher is better.

The Paterson-Stockmeyer method takes a value k, that is theoretically optimal when k = sqrt(2 * degree).
However, sometimes it is better to try other values of k (e.g. due to rounding and to trade off depth and cost).
This factor, let's call it f, is used to limit the candidate values of k that we try: [1, f * sqrt(2 * degree))."""
