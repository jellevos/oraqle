# Pareto fronts API
!!! warning
    In this version of Oraqle, the API is still prone to changes. Paths and names can change between any version.
    
If you are using depth-aware arithmetization, you will find that the compiler does not output one arithmetic circuit.
Instead, it outputs a Pareto front, which represents the best circuits that it could generate trading off two metrics:
The *multiplicative depth* and the *multiplicative size/cost*.
This page briefly explains the API for interfacing with these Pareto fronts.

## The abstract base class

??? info "Abstract ParetoFront"
    ::: oraqle.compiler.nodes.abstract.ParetoFront
        options:
            heading_level: 3

## Depth-size and depth-cost fronts

