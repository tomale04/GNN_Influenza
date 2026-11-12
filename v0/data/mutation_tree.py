"""
mutation_tree.py
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
    "flag_negative_value", "flag_bad_frequency"
    ]

COLLAPSE = {
    "ha_muts": 0,
}

class MutationTree():
    """
    A mutation tree object. Stored as a NetworkX tree.

    === Attributes ===
        cutoff_date = Date representing the forecast origin/data cutoff.
        In the format 'YYYY-MM-DD'.
        
        self.inFile = Path to raw phylogenetic tree (the Nextstrain Auspice 
        JSON file).

        self.outFile = Path specifying location to store the mutation tree.

        self.flagFile = Path to node and edge quality flags.

        self.node_flags = The flag file in a nice format (a list of dicts).

        self.phylo_tree = A NetworkX tree object representing the raw
        phylogenetic tree from inFile.

        self.pruned_phyle_tree = The phylogenetic tree which has been pruned
        based on data quality.

        self.mutation_tree = The mutation tree constructed by collapsing
        zero-mutation edges in the pruned phylogenetic tree.

        self.zero_mutation_edges = The collection of edges along which there
        are no mutations.

        self.collapsed_node_members = A dictionary where the keys are the nodes 
        in the pruned phylogenetic tree. The value at a given node is the set of
        nodes which were collapsed into it upon collapsing zero-mutation edges.

        gene_segments = List containing the sequence regions in which mutations 
        will be counted. Edges will be collapsed based on these segments.

        self.sequence_segments = Segments of the amino acid or nucleotide
        sequence to be included as node features in the mutation tree.
    """
    def __init__(self, cutoff_date, inFile, outFile, 
                flagFile, rootFile, gene_segments,
                sequence_segments=('HA1', 'HA2')):
        """
        Inputs:
            cutoff_date = Date representing the forecast origin/data cutoff.
            In the format 'YYYY-MM-DD'.

            inFile = Path to raw phylogenetic tree.

            outFile = Path specifying location to store the mutation tree.

            flagFile = Path to node and edge quality flags.

            rootFile = Path too file containing sequence information about
            the root node.
            
            gene_segments = List containing the sequence regions in which
            mutations will be counted.

            sequence_segments = Segments of the amino acid or nucleotide
            sequence to be included as node features in the mutation tree.

            self.sampled_tip_ids = The IDs of the sampled tips.
        
        Returns:
            None
        """

        self.cutoff_date = cutoff_date

        self.inFile = Path(inFile)
        self.outFile = Path(outFile)
        self.flagFile = Path(flagFile)
        self.rootFile = Path(rootFile)

        self.gene_segments = gene_segments
        self.sequence_segments = sequence_segments

        self.node_flags = None

        self.phylo_tree = None
        self.pruned_phylo_tree = None

        self.ref_sequence = None

        self.mutation_tree = None

        self.zero_mutation_edges = None
        self.collapsed_node_members = {}

        self.sampled_tip_ids = None

        return

    def construct(self):
        """
        Construct the actual mutation tree. This method should initialize
        self.phylo_tree, self.mutation_tree, and self.node_flags.
        """

        # Construct the initial phylogenetic tree and add the node flags.
        tree_json = self._load_json(self.inFile)
        self.node_flags = self._load_flags(self.flagFile)
        self.phylo_tree = self._construct_phylo_tree(tree_json)

        self._identify_sampled_tips(self.phylo_tree)

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

        self.mutation_tree = self._collapse_edges(
            G=self.pruned_phylo_tree
        )

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

    def _identify_sampled_tips(self, G):
        """
        Identify and store the terminal nodes of the original phylogenetic tree.

        Inputs:
            G: Original, unpruned phylogenetic tree.
        """

        self.sampled_tip_ids = {
            node
            for node in G.nodes
            if G.out_degree(node) == 0
        }

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
                    current, path_attributes, pruned_nodes = stack.pop()

                    # If this node should be pruned, we still need to check if
                    # its descendants need to be pruned. 
                    if current in nodes_to_prune:
                        for child in reversed(list(G.successors(current))):
                            edge_attrs = deepcopy(G.edges[current, child])

                            stack.append((
                                child,
                                path_attributes + [edge_attrs],
                                pruned_nodes + [current]
                            ))

                    else:
                        H.add_edge(
                            ancestor,
                            current,
                            attributes={
                                'mutations': self._concatenate_mutations(path_attributes)
                            },
                            pruned_nodes=pruned_nodes
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

    def _concatenate_mutations(self, path_attributes):
        """
        Upon pruning a node, update the specified mutation annotations along
        the resulting branch by concatenating the mutations into the pruned
        node with the mutations from the pruned node to its child.
        """

        aggregated = {}

        for edge_attrs in path_attributes:
            mutations = edge_attrs.get('attributes', {}).get('mutations', {})

            for seq_type, seq_mutations in mutations.items():
                if seq_type not in aggregated:
                    aggregated[seq_type] = []

                aggregated[seq_type] += seq_mutations

        return aggregated
        
    def _collapse_edges(self, G):
        """
        Collapse zero-mutation edges based on mutations in <gene_seqments>.

        Inputs:
            G = The phylogenetic tree.

            gene_seqments = List containing the sequence regions in which
            mutations will be counted.

        Returns:
            phylo_tree_edge_collapsed = The phylogenetic tree after edges
            have been collapsed.
        """

        zero_mutation_edges = self._get_zero_mutation_edges(
            G=G,
            gene_segments=self.gene_segments
        )

        self.zero_mutation_edges = zero_mutation_edges

        roots = [node for node, degree in G.in_degree() if degree == 0]
        root = roots[0]

        # Make a copy of G
        H = nx.DiGraph()
        H.graph.update(deepcopy(G.graph))

        H.add_node(root)
        self.collapsed_node_members = {root: [root]}

        self._traverse_and_collapse(
            G=G, H=H, 
            original_node=root, 
            collapsed_node=root, 
            zero_mutation_edges=zero_mutation_edges,
            path_attributes=[]
        )

        self._aggregate_node_features(G, H)

        return H

    def _get_zero_mutation_edges(self, G, gene_segments):
        """
        Returns a list of edges along which there are no mutations in
        <gene_segments>.

        Inputs:
            G = The phylogenetic tree.

            gene_segments = List containing the sequence regions in which
            mutations will be counted.

        Returns:
            zero_mutation_edges = A set of edges along which there 
            are no mutations in <gene_segments>.
        """

        zero_mutation_edges = set()

        print(G.edges(data=True))
        for parent, child, edge_data in G.edges(data=True):
            edge = (parent, child)

            mutations = edge_data.get(
                'attributes', {}
            ).get('mutations', {})
            
            no_mutations = not any(
                mutations.get(segment, [])
                for segment in gene_segments
            )

            if no_mutations:
                zero_mutation_edges.add(edge)

        return zero_mutation_edges

    def _traverse_and_collapse(
            self, G, H, original_node, collapsed_node, 
            zero_mutation_edges, path_attributes=None
    ):
        """
        Recursively traverse G and build H by contracting zero-mutation edges.

        Inputs:
            G = Original phylogenetic tree being traversed.

            H = New tree being constructed.

            original_node = Current node in G being visited.

            collapsed_node = Node in H to which descendants are attached. If
            original_node was reached through a zero-mutation path, collapsed_node
            points to the "oldest" ancestor along this path.

            zero_mutation_edges = List of edges along which there are no
            specified mutations

        """

        if path_attributes is None:
            path_attributes = []
        
        for child in G.successors(original_node):

            edge_attrs = deepcopy(G.edges[original_node, child])

            if (original_node, child) in zero_mutation_edges:

                self.collapsed_node_members[collapsed_node].append(child)

                self._traverse_and_collapse(
                    G, H, child, collapsed_node, 
                    zero_mutation_edges,
                    path_attributes + [edge_attrs]
                )

            else:
                H.add_node(child)

                self.collapsed_node_members[child] = [child]

                accumulated_mutations = self._concatenate_mutations(
                    path_attributes=path_attributes + [edge_attrs]
                )

                H.add_edge(
                    collapsed_node,
                    child,
                    attributes={'mutations': accumulated_mutations}
                )

                self._traverse_and_collapse(
                    G, H, child, child,
                    zero_mutation_edges
                )
        
        return

    def _aggregate_node_features(self, G, H):
        """
        Aggregate and assign node features . A list of
        chosen node features and their descriptions is included in the report.
        
        Returns:
            mutation_tree_feat = The mutation tree with node features added.
        """

        genotype_groups = self._group_sampled_tips_by_genotype(G)

        for mutation_node, component_nodes in self.collapsed_node_members.items():

            sequences = self._aggregate_sequences(
                G=G,
                component_nodes=component_nodes
            )

            representative = component_nodes[0]

            genotype = self._get_genotype_key(
                G.nodes[representative]['sequences']
            )

            genotype_tips = genotype_groups.get(genotype, [])

            H.nodes[mutation_node].update({
                'sequences': sequences,

                'frequency': self._aggregate_genotype_frequency(
                    G=G,
                    tips=genotype_tips
                ),

                'collection_times': self._aggregate_collection_times(
                    G=G,
                    tips=genotype_tips
                ),

                'clade': self._aggregate_clade(
                    G=G, 
                    component_nodes=component_nodes
                ),

                'lbi': self._recalculate_lbi(
                    G=G, 
                    H=H,
                    component_nodes=component_nodes
                ),

                'component_nodes': component_nodes
            })

        return H

    def _group_sampled_tips_by_genotype(self, G):
        """
        Group terminal nodes of the phylogenetic tree by amino-acid
        genotype. Each genotype is identified by its amino-acid 
        sequences across self.gene_segments.
        
        Returns:
            genotype_groups = Dictionary mapping genotype sequences
            to lists of tip IDs.
        """

        genotype_groups = {}

        for node in G.nodes:

            if node not in self.sampled_tip_ids:
                continue

            sequences = G.nodes[node]['sequences']
            genotype = self._get_genotype_key(sequences)

            genotype_groups.setdefault(genotype, []).append(node)
        
        return genotype_groups

    def _get_genotype_key(self, sequences):
        """
        Convert the dictionary of sequences into a tuple containing where
        each entry is one of the dictionary values (actual sequence).

        Inputs:
            sequences = dictionary containing amino acid/nucleotide seuquences.
        """

        return tuple(
            str(sequences[segment])
            for segment in self.gene_segments
        )   

    def _aggregate_sequences(self, G, component_nodes):
        """
        Retrieve the amino-acid sequences represented by a collapsed
        mutation-tree node.
        
        Inputs:
            G = Original phylogenetic tree.
            component_nodes = Original phylogenetic tree nodes which have
            been collapsed with the zero-mutation edges.
        """

        if not component_nodes:
            raise ValueError("component_nodes is empty.")

        representative = component_nodes[0]
        sequences = G.nodes[representative]['sequences']

        for node in component_nodes[1:]:
            node_sequences = G.nodes[node]['sequences']

            for segment in self.gene_segments:
                if sequences[segment] != node_sequences[segment]:
                    raise ValueError(
                        f"Collapsed component containing {representative} "
                        f"and {node} has inconsistent sequences in "
                        f"segment {segment}."
                    )

        return {
            segment: str(sequences[segment])
            for segment in self.sequence_segments
        }

    def _aggregate_genotype_frequency(self, G, tips):
        return

    def _aggregate_collection_times(self, G, tips):
        """
        Collect the unique collection dates associated with a genotype.
        
        Inputs:
            G = Pruned phylogenetic tree.
        
            tips = Sampled tip IDs belonging to the genotype.
        
        Outputs:
            collection_times = Sorted list of unique collection dates.

        Note that this list will be empty if this genotype has no sampled
        observations (tips).
        """

        collection_times = set()

        for tip in tips:
            num_date = G.nodes[tip].get('attributes', {}).get('num_date')

            if isinstance(num_date, dict):
                num_date = num_date.get('value')

            if num_date is None:
                continue

            collection_times.add(float(num_date))

        return sorted(collection_times)

    def _aggregate_clade(self, G, component_nodes):
        return

    def _recalculate_lbi(self, G, H, component_nodes):
        """
        Re-calculate the Local Branching Index for a given node in the
        mutation tree."""
        return

    def _get_edge_length(self, G, parent, child):
        """
        Use the values of the <div> node attribute to get the edge length
        in the mutation tree.
        
        Note that the edge lengths should be roughly the same as we are
        collapsing zero-mutation edges. However, we only collapse based
        on mutations in self.gene_segments, and mutations in other segments
        are included in <div>.
        """
        parent_div = G.nodes[parent]['attributes']['div']
        child_div = G.nodes[child]['attributes']['div']

        return child_div - parent_div
        
    def _assign_node_features(self):
        """
        Assign node features to each node in the mutation tree.
        """

        # TODO: Implement this method.

        return

    def _assign_edge_features(self):
        """
        Assign edge features to each edge in the tree.
        """

        # TODO: Implement this method.

        return

    # Any sort of validation we could do?

    def write_output(self):
        """
        Store the mutation tree at the specified location.
        """

        # TODO: Implement this method.

        return
        
