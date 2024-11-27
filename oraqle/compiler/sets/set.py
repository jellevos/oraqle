from typing import Type
from galois import FieldArray
from oraqle.compiler.nodes.leafs import Input


class AbstractSet(Input):
    pass


class Set(AbstractSet):
    # TODO: This should have a certain (runtime-available) type for the elements
    def __init__(self, name: str, gf: type[FieldArray], universe_size: int) -> None:
        super().__init__(name, gf)
        self._universe_size = universe_size

# if __name__ == "__main__":
#     Set()
