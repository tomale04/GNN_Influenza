"""
mutation_graph.py
=================

Description to come...
"""

import networkx as nx
import json
import csv
from pathlib import Path
from copy import deepcopy

# --------------------------------------------------------------------------- #
# CONSTANTS
# --------------------------------------------------------------------------- #

# Chose these flags based on completeness and what seems important.
FLAGS = [
    "flag_qc_bad", "flag_qc_mediocre", 
    "flag_negative_value", "flag_data_out_of_range",
    "flag_bad_frequency"
    ]

COLLAPSE = {
    "ha_muts": 0,
}

class MutationGraph():
    """
    A mutation graph object. Stored as a NetworkX graph.

    === Attributes ===
        self.inFile = Path to raw phylogenetic tree.
        self.outFile = Path specifying location to store the mutation graph.
        self.flagFile = Path to node and edge quality flags.
        self.node_flags = The flag file in a nice format (a list of dicts).
        phylo_tree = A NetworkX graph object representing the raw
        phylogenetic tree from inFile.
        pruned_phyle_tree = The phylogenetic tree which has been pruned
        based on data quality.
        mutation_graph = The mutation graph constructed by collapsing
        zero-mutation edges in the pruned phylogenetic tree.
    """
    def __init__(self, inFile, outFile, flagFile, rootFile):
        """
        Inputs:
            inFile = Path to raw phylogenetic tree.
            outFile = Path specifying location to store the mutation graph.
            flagFile = Path to node and edge quality flags.
            rootFile = Path too file containing sequence information about
            the root node.
        
        Returns:
            None
        """
        self.inFile = Path(inFile)
        self.outFile = Path(outFile)
        self.flagFile = Path(flagFile)
        self.rootFile = Path(rootFile)

        self.node_flags = None

        self.phylo_tree = None
        self.pruned_phylo_tree = None

        self.ref_sequence = None

        self.mutation_graph = None

        return

    def construct(self):
        """
        Construct the actual mutation graph. This method should initialize
        self.phylo_tree, self.mutation_graph, and self.node_flags.
        """

        # Construct the initial phylogenetic tree and add the node flags.
        tree_json = self._load_json(self.inFile)
        self.node_flags = self._load_flags(self.flagFile)
        self.phylo_tree = self._construct_phylo_tree(tree_json)

        # Add sequences as individual node attributes.
        roots = [node for node, degree in self.phylo_tree.in_degree() if degree == 0]
        root = roots[0]
        self.ref_sequence = self._load_json(self.rootFile)

        self.phylo_tree = self._reconstruct_sequences(
            G=self.phylo_tree,
            root=root,
            root_sequence=self.ref_sequence
        )

        # Prune the flagged nodes and recalculate attributes.
        self.pruned_phylo_tree = self._prune_flagged_nodes(
            G=self.phylo_tree
        )

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

        Inputs:
            tree_json = JSON file containing the raw phylogenetic tree data.
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

            notes = [
                note.strip()
                for note in flag_row.get("notes", "").split(" | ")
                if note.strip()
            ]

            missing_attributes = [
                note.split(":", 1)[1]
                for note in notes
                if note.startswith("missing:")
            ]

            placeholder_attributes = [
                note.split(":", 1)[1].split("=", 1)[0]
                for note in notes
                if note.startswith("placeholder:")
            ]

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
                flag_reasons=flag_row.get("flag_reasons", ""),
                notes=notes,
                missing_attributes=missing_attributes,
                placeholder_attributes=placeholder_attributes
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

    def _reconstruct_sequences(self, G, root, root_sequence):
        """
        Construct the actual amino acid sequences for each internal node
        given the root amino acid (or nucleotide) sequence and a specified 
        list of mutations along each branch. This must be applied to the 
        phylogenetic tree BEFORE pruning.
        """

        tree_json = self._load_json(self.inFile)

        sequences = {
            root: self._apply_mutations(
                parent_sequence=root_sequence,
                mutations=tree_json['tree']['branch_attrs']['mutations']
            )
        }

        stack = [root]

        while stack:
            parent = stack.pop()

            for child in G.successors(parent):

                mutations = G.edges[(parent, child)]['attributes']['mutations']

                child_sequence = self._apply_mutations(
                    parent_sequence=sequences[parent],
                    mutations=mutations
                )

                sequences[child] = child_sequence

                stack.append(child)

        nx.set_node_attributes(G, sequences, 'sequences')

        return G

    def _apply_mutations(self, parent_sequence, mutations):

        child_sequence = {}

        for seq_type in parent_sequence.keys():

            if seq_type not in mutations.keys():
                child_sequence[seq_type] = parent_sequence[seq_type]

            else:
                child_seq = list(parent_sequence[seq_type])

                for mutation in mutations[seq_type]:

                    ancestral = mutation[0]
                    position = int(mutation[1:-1]) - 1
                    derived = mutation[-1]

                    observed = child_seq[position]

                    if observed != ancestral:
                        raise ValueError(
                            f"Mutation mismatch: {mutation}, "
                            f"expected {ancestral} at position {position + 1}, "
                            f"found {observed}"
                        )

                    child_seq[position] = derived

                child_sequence[seq_type] = "".join(child_seq)

        return child_sequence

    def _prune_flagged_nodes(self, G):
        """
        Prune flagged nodes from the phlogenetic tree. This method should
        initialize self.pruned_phylo_tree.
        """

        nodes_to_prune = self._get_nodes_to_prune(G)

        roots = [node for node, degree in G.in_degree() if degree == 0]
        root = roots[0]

        if root in nodes_to_prune:
            raise ValueError("The root is flagged.")

        # Make a copy of G
        H = nx.DiGraph()
        H.graph.update(deepcopy(G.graph))

        for node, attrs in G.nodes(data=True):
            if node not in nodes_to_prune:
                H.add_node(node, **deepcopy(attrs))

        for ancestor in H.nodes:

            stack = []

            for child in reversed(list(G.successors(ancestor))):
                edge_attrs = deepcopy(G.edges[ancestor, child])

                # Add the node we are about to visit, edge attributes from the
                # previous branch, and flagged nodes encountered along the path.
                stack.append((child, [edge_attrs], []))

                while stack:
                    current, path_attributes, contracted_nodes = stack.pop()

                    # If this node should be pruned, we still need to check that
                    # its descendants do not need to be pruned. 
                    if current in nodes_to_prune:
                        for child in reversed(list(G.successors(current))):
                            edge_attrs = deepcopy(G.edges[current, child])

                            stack.append((
                                child,
                                path_attributes + [edge_attrs],
                                contracted_nodes + [current]
                            ))

                    else:
                        H.add_edge(
                            ancestor,
                            current,
                            path_attributes=path_attributes,
                            contracted_nodes=contracted_nodes
                        )

        self.pruned_phylo_tree = H

        return H

    def _get_nodes_to_prune(self, G):
        """
        Returns a list of nodes in a phylogenetic tree which should be
        pruned.

        Inputs:
            G = NetworkX object representing a phylogenetic tree. For our
            pruposes this is self.phylo_tree.
        """

        return {
            node
            for node, attrs in G.nodes(data=True)
            if any(attrs["quality_flags"].get(flag, False)
                for flag in FLAGS)
        }
        
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
        
