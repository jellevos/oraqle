"""This module contains classes for representing circuits."""
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

from fhegen.bgv import logqP
from fhegen.util import estsecurity
from galois import FieldArray

from oraqle.compiler.graphviz import DotFile
from oraqle.compiler.instructions import ArithmeticProgram, OutputInstruction
from oraqle.compiler.nodes.abstract import ArithmeticNode, Node


class Circuit:
    """Represents a circuit over a fixed finite field that can be turned into an arithmetic circuit. Behind the scenes this is a directed acyclic graph (DAG). The circuit only has references to the outputs."""

    def __init__(self, outputs: List[Node]):
        """Initialize a circuit with the given `outputs`."""
        assert len(outputs) > 0
        self._outputs = outputs
        self._gf = outputs[0]._gf

    def evaluate(self, actual_inputs: Dict[str, FieldArray]) -> List[FieldArray]:
        """Evaluates the circuit with the given named inputs.
        
        This function does not error if it is given more inputs than necessary, but it will error if one is missing.
        
        Returns:
            Evaluated output in plain text.
        """
        assert all(isinstance(value, self._gf) for value in actual_inputs.values())

        actual_outputs = [output.evaluate(actual_inputs) for output in self._outputs]
        self._clear_cache()

        return actual_outputs

    def to_graph(self, file_name: str):
        """Saves a DOT file representing the circuit as a graph at the given `file_name`."""
        graph_builder = DotFile()

        for output in self._outputs:
            graph_builder.add_link(
                output.to_graph(graph_builder),
                graph_builder.add_node(label="Output", shape="plain"),
            )
        self._clear_cache()

        graph_builder.to_file(file_name)

    def to_pdf(self, file_name: str):
        """Saves a PDF file representing the circuit as a graph at the given `file_name`."""
        with tempfile.NamedTemporaryFile(suffix=".dot", delete=False) as dot_file:
            self.to_graph(dot_file.name)

        subprocess.run(["dot", "-Tpdf", dot_file.name, "-o", file_name], check=True)

    def to_svg(self, file_name: str):
        """Saves an SVG file representing the circuit as a graph at the given `file_name`."""
        with tempfile.NamedTemporaryFile(suffix=".dot", delete=False) as dot_file:
            self.to_graph(dot_file.name)

        subprocess.run(["dot", "-Tsvg", dot_file.name, "-o", file_name], check=True)

    def display_graph(self, metadata: Optional[dict] = None):
        """Displays the circuit in a Python notebook."""
        with tempfile.NamedTemporaryFile(suffix=".dot", delete=False) as dot_file:
            self.to_graph(dot_file.name)

        with open(dot_file.name, encoding="utf8") as file:
            file_content = file.read()

        import graphviz
        from IPython.display import display_png

        src = graphviz.Source(file_content)
        display_png(src, metadata=metadata)

    def eliminate_subexpressions(self):
        """Perform semantic common subexpression elimination on all outputs."""
        for output in self._outputs:
            output.eliminate_common_subexpressions({})

    def is_equivalent(self, other: object) -> bool:
        """Returns whether the two circuits are semantically equivalent.
        
        False positives do not occure but false negatives do.
        """
        if not isinstance(other, self.__class__):
            return False

        return all(out1.is_equivalent(out2) for out1, out2 in zip(self._outputs, other._outputs))

    def arithmetize(self, strategy: str = "best-effort") -> "ArithmeticCircuit":
        """Arithmetizes this circuit by calling arithmetize on all outputs.
        
        This replaces all high-level operations with arithmetic operations (constants, additions, and multiplications).
        The current implementation only aims at reducing the total number of multiplications.

        Returns:
            An equivalent arithmetic circuit with low multiplicative size.
        """
        arithmetic_circuit = ArithmeticCircuit(
            [output.arithmetize(strategy).to_arithmetic() for output in self._outputs]
        )
        # FIXME: Also call to_arithmetic
        arithmetic_circuit._clear_cache()

        return arithmetic_circuit

    def arithmetize_depth_aware(
        self, cost_of_squaring: float = 1.0
    ) -> List[Tuple[int, int, "ArithmeticCircuit"]]:
        """Perform depth-aware arithmetization on this circuit.
        
        !!! failure
            The current implementation only supports circuits with a single output.
    
        This function replaces high-level nodes with arithmetic operations (constants, additions, and multiplications).
        
        Returns:
            A list with tuples containing the multiplicative depth, the multiplicative cost, and the generated arithmetization from low to high depth.
        """
        assert len(self._outputs) == 1
        assert cost_of_squaring <= 1.0

        front = []
        for depth, size, node in self._outputs[0].arithmetize_depth_aware(cost_of_squaring):
            arithmetic_circuit = ArithmeticCircuit([node])
            arithmetic_circuit._clear_cache()
            front.append((depth, size, arithmetic_circuit))

        arithmetic_circuit._clear_cache()
        return front

    def _clear_cache(self):
        already_cleared = set()
        for output in self._outputs:
            output.clear_cache(already_cleared)


helib_preamble = """
#include <iostream>
#include <map>
#include <string>
#include <chrono>

#include <helib/helib.h>

typedef helib::Ptxt<helib::BGV> ptxt_t;
typedef helib::Ctxt ctxt_t;

std::map<std::string, int> input_map;

void parse_arguments(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string argument(argv[i]);
        size_t pos = argument.find('=');
        if (pos != std::string::npos) {
            std::string key = argument.substr(0, pos);
            int value = std::stoi(argument.substr(pos + 1));
            input_map[key] = value;
        }
    }
}

int extract_input(const std::string& name) {
    if (input_map.find(name) != input_map.end()) {
        return input_map[name];
    } else {
        std::cerr << "Error: " << name << " not found" << std::endl;
        return -1;
    }
}

int main(int argc, char* argv[]) {
    // Parse the inputs
    parse_arguments(argc, argv);
"""

helib_keygen = """
    // Generate keys
    helib::SecKey secret_key(context);
    secret_key.GenSecKey();
    helib::addSome1DMatrices(secret_key);
    const helib::PubKey& public_key = secret_key;
"""

helib_postamble = """
    return 0;
}
"""


class ArithmeticCircuit(Circuit):
    """Represents an arithmetic circuit over a fixed finite field, so it only contains arithmetic nodes."""

    _outputs: List[ArithmeticNode]

    def multiplicative_depth(self) -> int:
        """Returns the multiplicative depth of the circuit."""
        depth = max(output.multiplicative_depth() for output in self._outputs)
        self._clear_cache()

        return depth

    def multiplicative_size(self) -> int:
        """Returns the multiplicative size (number of multiplications) of the circuit."""
        multiplications = set().union(*(output.multiplications() for output in self._outputs))
        size = len(multiplications)

        return size

    def multiplicative_cost(self, cost_of_squaring: float) -> float:
        """Returns the multiplicative cost of the circuit."""
        multiplications = set().union(*(output.multiplications() for output in self._outputs))
        squarings = set().union(*(output.squarings() for output in self._outputs))
        cost = len(multiplications) - len(squarings) + cost_of_squaring * len(squarings)

        return cost

    def generate_program(self) -> ArithmeticProgram:
        """Returns an arithmetic program for this arithmetic circuit."""
        # Reset the parent counts
        for output in self._outputs:
            output.reset_parent_count()

        # Count the parents
        for output in self._outputs:
            output.count_parents()

        # Reset the cache for instruction writing
        self._clear_cache()

        # Write the instructions
        instructions = []
        stack_occupied = []

        stack_counter = 0
        for output in self._outputs:
            output_index, stack_counter = output.create_instructions(
                instructions, stack_counter, stack_occupied
            )
            instructions.append(OutputInstruction(output_index))

        # Reset the cache for future operations
        self._clear_cache()

        return ArithmeticProgram(instructions, len(stack_occupied), self._gf)

    def summands_between_multiplications(self) -> int:
        """Computes the maximum number of summands between two consecutive multiplications in this circuit.

        !!! failure
            This currently returns the hardcoded value 10
        
        Returns:
            The highest number of summands between two consecutive multiplications
        """
        # FIXME: This is currently hardcoded
        return 10

    def _generate_helib_params(self) -> Tuple[str, Tuple[int, int, int, int]]:
        # Returns the code, along with (m, r, bits, c)
        multiplicative_depth = self.multiplicative_depth()
        summands_between_mults = self.summands_between_multiplications()

        # This code is adapted from fhegen: https://github.com/Crypto-TII/fhegen
        # It was written by Johannes Mono, Chiara Marcolla, Georg Land, Tim Güneysu, and Najwa Aaraj

        ops = {
            "model": "OpenFHE",
            "muls": multiplicative_depth + 1,
            "const": True,
            "rots": 0,
            "sums": summands_between_mults,
        }

        sdist = "Ternary"
        sigma = 3.19
        ve = sigma * sigma
        vs = {"Ternary": 2 / 3, "Error": ve}[sdist]
        b_args = {
            "m": 4,
            "t": self._gf.characteristic,
            "D": 6,
            "Vs": vs,
            "Ve": ve,
        }  # We will loop over increasing m to find a suitable value
        kswargs = {"method": "Hybrid-RNS", "L": multiplicative_depth + 1, "beta": 2**10, "omega": 3}

        while True:
            logq, logp = logqP(ops, b_args, kswargs, sdist)
            log = sum(logq) + logp if logp else sum(logq)
            if logp and estsecurity(b_args["m"], log, sdist) >= 128:
                break

            b_args["m"] <<= 1

        # TODO: This is a workaround
        if self._gf.characteristic == 2:
            b_args["m"] -= 1

        sec = estsecurity(b_args["m"], sum(logq) + logp, sdist)
        assert sec >= 128

        return f"""
    // Set up the HE parameters
    unsigned long p = {self._gf.characteristic};
    unsigned long m = {b_args["m"]};
    unsigned long r = 1;
    unsigned long bits = {sum(logq)};
    unsigned long c = 3;
    helib::Context context = helib::ContextBuilder<helib::BGV>()
        .m(m)
        .p(p)
        .r(r)
        .bits(bits)
        .c(c)
        .build();
""", (b_args["m"], 1, sum(logq), 3)

    def generate_code(
        self,
        filename: str,
        iterations: int = 1,
        measure_time: bool = False,
        decrypt_outputs: bool = False,
    ) -> Tuple[int, int, int, int]:
        """Generates an HElib implementation of the circuit.
        
        If decrypt_outputs is True, prints the decrypted output.
        Otherwise, it prints whether the ciphertext has noise budget remaining (i.e. it is correct with high probability).

        !!! note
            Decryption is part of the measured run time.

        Args:
            filename: Test
            iterations: Number of times to run the circuit
            measure_time: Whether to output a measurement of the total run time
            decrypt_outputs: Whether to print the decrypted outputs, or to simply check if there is noise budget remaining

        Returns:
            Parameters that were chosen: (ring dimension m, Hensel lifting = 1, bits in the modchain, columns in key switching = 3).
        """
        from oraqle.compiler.instructions import InputInstruction

        # Generate HElib code
        with open(filename, "w", encoding="utf8") as file:
            # Write start of file and parameters
            file.write(helib_preamble)
            param_code, params = self._generate_helib_params()
            file.write(param_code)
            file.write("\n")
            file.write(helib_keygen)
            file.write("\n")

            # Encrypt the inputs
            program = self.generate_program()
            inputs = [
                instruction._name
                for instruction in program._instructions
                if isinstance(instruction, InputInstruction)
            ]
            file.write("\t// Encrypt the inputs\n")
            for input in inputs:
                file.write(
                    f'\tstd::vector<long> vec_{input}(1, extract_input("{input}"));\n\tptxt_t ptxt_{input}(context, vec_{input});\n\tctxt_t ciph_{input}(public_key);\n\tpublic_key.Encrypt(ciph_{input}, ptxt_{input});\n'
                )
            file.write("\n")

            # If timing is enabled, start the timer
            if measure_time:
                file.write("\tauto start = std::chrono::high_resolution_clock::now();\n")
                file.write("\n")

            # If we perform multiple iterations, wrap in a for loop
            if iterations > 1:
                file.write(f"\tfor (int i = 0; i < {iterations}; i++) {{\n")

            # Write the actual instructions
            file.write("\t// Perform the actual circuit\n")
            file.write(
                "\n".join(
                    f"\t{line}" for line in program.generate_code(decrypt_outputs).splitlines()
                )
            )
            file.write("\n")

            # If we perform multiple iterations, close the for loop
            if iterations > 1:
                file.write("\t}\n")

            # If timing is enabled, stop the timer
            if measure_time:
                file.write("\n")
                file.write("\tauto end = std::chrono::high_resolution_clock::now();\n")
                file.write("\tstd::chrono::duration<double> elapsed = end - start;\n")
                file.write("\tstd::cout << elapsed.count() << std::endl;")
                file.write("\n")

            # Finish the file
            file.write(helib_postamble)

            return params


if __name__ == "__main__":
    from galois import GF

    from oraqle.compiler.circuit import Circuit
    from oraqle.compiler.nodes.leafs import Input

    gf = GF(7)

    x = Input("x", gf)
    y = Input("y", gf)

    arithmetic_circuit = Circuit([x < y]).arithmetize()
    arithmetic_circuit.generate_code("main.cpp", iterations=10, measure_time=True)
