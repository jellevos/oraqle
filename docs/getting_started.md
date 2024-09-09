# Getting started
In 5 minutes, this page will guide you through how to install oraqle, how to specify high-level programs, and how to arithmetize your first circuit!

## Installation
Simply install the most recent version of the Oraqle compiler using:
```
pip install oraqle
```

We use continuous integration to test every build of the Oraqle compiler on Windows, MacOS, and Unix systems.
If you do run into problems, feel free to [open an issue on GitHub]()!

## Specifying high-level programs
Let's start with importing `galois`, which represents our plaintext algebra.
We will also immediately import the relevant oraqle classes for our little example:
```python3
from galois import GF

from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.leafs import Input
```

For this example, we will use 31 as our plaintext modulus. This algebra is denoted by `GF(31)`.
Let's create a few inputs that represent elements in this algebra:
```python3
gf = GF(31)

x = Input("x", gf)
y = Input("y", gf)
z = Input("z", gf)
```

We can now perform some operations on these elements, and they do not have to be arithmetic operations!
For example, we can perform equality checks or comparisons:
```
comparison = x < y
equality = y == z
both = comparison & equality
```

While we have specified some operations, we have not yet established this as a circuit. We will do so now:
```python3
circuit = Circuit([both])
```

And that's it! We are done specifying our first high-level circuit.
As you can see this is all very similar to writing a regular Python program.
If you want to visualize this high-level circuit before we continue with arithmetizing it, you can run the following (if you have graphviz installed):
```python3
circuit.to_pdf("high_level_circuit.pdf")
```

!!! tip
    If you do not have graphviz installed, you can instead call:
    ```python3
    circuit.to_dot("high_level_circuit.dot")
    ```
    After that, you can copy the file contents to [an online graphviz viewer](https://dreampuf.github.io/GraphvizOnline)!

## Arithmetizing your first circuit
At this point, arithmetization is a breeze, because the oraqle compiler takes care of these steps.
We can create an arithmetic circuit and visualize it using the following snippet:
```python3
arithmetic_circuit = circuit.arithmetize()
arithmetic_circuit.to_pdf("arithmetic_circuit.pdf")
```

You will notice that it's quite a large circuit. But how large is it exactly?
This is a question that we can ask to the oraqle compiler:
```python3
print("Depth:", arithmetic_circuit.multiplicative_depth())
print("Size:", arithmetic_circuit.multiplicative_size())
print("Cost:", arithmetic_circuit.multiplicative_cost(0.7))
```

In the last line, we asked the compiler to output the multiplicative cost, considering that squaring operations are cheaper than regular multiplications.
We weighed this cost with a factor 0.7.

Now that we have an arithmetic circuit, we can use homomorphic encryption to evaluate it!
If you are curious about executing these circuits for real, consider reading [the code generation tutorial](tutorial_running_exps.md).

!!! warning
    There are many homomorphic encryption libraries that do not support plaintext moduli that are not NTT-friendly. The plaintext modulus we chose (31) is not NTT-friendly.
    In fact, only very few primes are NTT-friendly, and they are somewhat large. This is why, right now, the oraqle compiler only implements code generation for HElib.
    HElib is (as far as we are aware) the only library that supports plaintext moduli that are not NTT-friendly.
