"""This module contains classes and functions for visualizing circuits using graphviz."""
from typing import Dict, List, Sequence, Tuple

expensive_style = {"shape": "diamond"}


class DotFile:
    """A `DotFile` is a graph description format that can be rendered to e.g. PDF using graphviz."""
    
    def __init__(self):
        """Initialize an empty DotFile."""
        self._nodes: List[Dict[str, str]] = []
        self._links: List[Tuple[int, int, Dict[str, str]]] = []
        self._clusters: List[Tuple[Sequence[int], Dict[str, str]]] = []

    def add_node(self, **kwargs) -> int:
        """Adds a node to the file. The keyword arguments are directly put into the DOT file.

        For example, one can specify a label, a color, a style, etc...

        Returns:
            The identifier of this node in this `DotFile`.
        """
        node_id = len(self._nodes)
        self._nodes.append(kwargs)

        return node_id

    def add_link(self, from_id: int, to_id: int, **kwargs):
        """Adds an unformatted link between the nodes with `from_id` and `to_id`. The keyword arguments are directly put into the DOT file."""
        assert from_id != -1
        assert to_id != -1
        self._links.append((from_id, to_id, kwargs))

    def add_cluster(self, ids: Sequence[int], **kwargs):
        """Adds a cluster containings the nodes with the given IDs. The keyword arguments are directly put into the DOT file."""
        assert -1 not in ids
        self._clusters.append((ids, kwargs))

    def to_file(self, filename: str):
        """Writes the DOT file to the given filename as a directed graph called 'G'."""
        with open(filename, mode="w", encoding="utf-8") as file:
            file.write("digraph G {\n")
            file.write('forcelabels="true";\n')
            file.write("graph [nodesep=0.25,ranksep=0.6];")  # nodesep, ranksep

            # Write all the nodes
            for node_id, attributes in enumerate(self._nodes):
                transformed_attributes = ",".join(
                    [f'{key}="{value}"' for key, value in attributes.items()]
                )
                file.write(f"n{node_id} [{transformed_attributes}];\n")

            # Write all the links
            for from_id, to_id, attributes in self._links:
                if len(attributes) == 0:
                    file.write(f"n{from_id}->n{to_id};\n")
                else:
                    text = f"n{from_id}->n{to_id} ["
                    text += ",".join((f"{key}={value}" for key, value in attributes.items()))
                    text += "];\n"
                    file.write(text)

            # Write all the clusters
            for cluster_id, properties in enumerate(self._clusters):
                ids, attributes = properties
                text = f"subgraph cluster_{cluster_id} {{\n"
                for id in ids:
                    text += f"n{id};\n"
                text += "}"
                file.write(text)

            file.write("}\n")
