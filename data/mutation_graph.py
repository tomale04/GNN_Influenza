# This file will contain the code to construct the mutation graph using the processed data.
import networkx as nx
import json
import csv
from pathlib import Path

class MutationGraph():
    """
    A mutation graph object. Stored as a NetworkX graph.
    """
    def __init__(self, inFile, outFile, flagFile):
        """
        Inputs:
            inFile = Path to raw phylogenetic tree.
            outFile = Path specifying location to store the mutation graph.
            flagFile = Path to node and edge quality flags.
        
        Returns:
            None
        """
        self.inFile = Path(inFile)
        self.outFile = Path(outFile)
        self.flagFile = Path(flagFile)

        self.phylo_tree = None
        self.mutation_graph = None
        self.node_flags = None
        return

    def construct(self):
        """
        Construct the actual mutation graph. This method should initialize
        self.phylo_tree, self.mutation_graph, and self.node_flags.
        """
        tree_json = self._load_json(self.inFile)
        self.node_flags = self._load_flags(self.flagFile)

        self.phylo_tree = self._construct_phylo_tree(tree_json)

        self.mutation_graph = self._collapse_edges()
        self._assign_node_features()
        self._assign_edge_features()

    def _load_json(self, path):
        """Load a JSON file."""
        with open(path, "r") as f:
            return json.load(f)

    def _load_flags(self, path):
        """Load node quality flags from CSV."""
        with open(path, "r") as f:
            return list(csv.DictReader(f))

    def _construct_phylo_tree(self, tree_json):
        """Convert the Nextstrain JSON tree into a NetworkX graph. Add the 
        quality flags from self.flagFile as key-value attributes to each node.
        """
        G = nx.DiGraph()

        root = tree_json["tree"]

        flags_by_id = {
            row["node_id"]: row
            for row in self.node_flags
        }

        stack = [(root, None, "root")]

        while stack:
            node, parent_id, node_id = stack.pop()

            node_attrs = node.get("node_attrs", {}) or {}
            edge_attrs = node.get("branch_attrs", {}) or {}
            children = node.get("children", []) or []

            # Nextstrain has this weird structure of storing values of node attributes
            # in a weird nested dictionary like 'node_attr_key' : {'value': x}. This
            # line of code just condenses that to 'node_attr_key: x' instead.
            node_attrs = {
                key: value.get("value") if isinstance(value, dict) else value
                for key, value in node_attrs.items()
            }

            flag_row = flags_by_id.get(node_id, {})

            quality_flags = {
                key: value == "True"
                for key, value in flag_row.items()
                if key.startswith("flag_")
            }

            G.add_node(
                node_id,
                name=node.get("name"),
                attributes=node_attrs,
                quality_flags=quality_flags,
                any_flag=flag_row.get("any_flag") == "True",
                flag_reasons=flag_row.get("flag_reasons", "")
            )

            if parent_id is not None:
                G.add_edge(
                    parent_id, 
                    node_id,
                    attributes=edge_attrs
                )

            # Traverse child nodes in original order
            for i in range(len(children) - 1, -1, -1):
                child_id = f"{node_id}/{i}"
                stack.append((children[i], node_id, child_id))

        return G
        
    def _collapse_edges(self):
        """
        Collapse zero-mutation edges.
        """
        return

    def _assign_node_features(self):
        """
        Assign node features to each node in the graph.
        """
        return

    def _assign_edge_features(self):
        """
        Assign edge features to each edge in the graph.
        """
        return

    # Any sort of validation we could do?

    def write_output(self):
        """
        Store the mutation graph at the specified location.
        """
        return
        
