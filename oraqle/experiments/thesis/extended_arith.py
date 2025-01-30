from typing import Any
from galois import GF

from oraqle.compiler.boolean.bool import BooleanInput
from oraqle.compiler.circuit import Circuit
from oraqle.compiler.sets.bitset import BitSet, BitSetContainer
from oraqle.mpc.compilation import to_subscript
from oraqle.mpc.parties import PartyId


class ZpElement:

    def __init__(self, element: int, modulus: int) -> None:
        self._element = element % modulus
        self._modulus = modulus

    def __add__(self, other: "ZpElement") -> "ZpElement":
        assert self._modulus == other._modulus
        return ZpElement((self._element + other._element) % self._modulus, self._modulus)
    
    def __int__(self) -> int:
        return self._element
    
    def __eq__(self, value: object) -> bool:
        assert isinstance(value, (int, ZpElement))
        element = value._element if isinstance(value, ZpElement) else value
        return self._element == element
    

class Zp:

    def __init__(self, prime: int) -> None:
        self._characteristic = prime

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        assert len(args) == 1
        assert len(kwds) == 0
        return ZpElement(args[0], self._characteristic)

    @property
    def characteristic(self) -> int:
        return self._characteristic
    
    @property
    def order(self) -> int:
        return self._characteristic


# TODO: Add proper set intersection interface
if __name__ == "__main__":
    zp = Zp(2**252 + 27742317777372353535851937790883648493)
        
    # TODO: Consider immediately creating a bitset (container) using bitset params/set params
    party_count = 3
    universe = 10
    party_bitsets = []
    for party_id in range(party_count):
        bits = [BooleanInput(f"b{to_subscript(party_id + 1)},{to_subscript(i + 1)}", zp, {PartyId(party_id)}) for i in range(universe)]  # type: ignore
        bitset = BitSetContainer(bits)
        party_bitsets.append(bitset)

    intersection = BitSet.intersection(*party_bitsets)

    circuit = Circuit([intersection.contains_element(element) for element in [1]]) # range(1, universe + 1)
    circuit.to_pdf("mpsi_hl.pdf")

    arithmetic_circuit = circuit.arithmetize()
    arithmetic_circuit.to_pdf("mpsi_arith.pdf")

    extended_arithmetic_circuit = circuit.arithmetize_extended()
    extended_arithmetic_circuit.to_pdf("mpsi_extended.pdf")
