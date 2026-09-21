import os
import typing as ty
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ..common.models import (
    CorrelationResult,
    VariableClusteringResult,
    VariableSelectionResult,
)


class ConstellationGraphPlotter:
    """Renders the global 'Constellation' network graph highlighting anchor variables and cluster groupings.
    """

    def __init__(self, size_anchor_node: int = 700, size_regular_node: int = 200):
        """Initializes the constellation plotter.

        Args:
            size_anchor_node: Display size for MMD-selected anchor nodes (hat_S).
            size_regular_node: Display size for regular/augmented nodes.
        """
        self.size_anchor_node = size_anchor_node
        self.size_regular_node = size_regular_node
        # end def __init__

    def plot_constellation_graph(
        self,
        correlation_result: CorrelationResult,
        clustering_result: VariableClusteringResult,
        selection_result: VariableSelectionResult,
        path_output_image: str
    ) -> str:
        """Constructs a force-directed network graph and exports it as an image artifact.

        Args:
            correlation_result: CorrelationResult containing edges and variables.
            clustering_result: VariableClusteringResult containing cluster assignments.
            selection_result: VariableSelectionResult containing anchor variable indices.
            path_output_image: Target file path to write image (.png or .pdf).

        Returns:
            Absolute file path to the saved graph figure.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path_output_image)), exist_ok=True)

        graph_network = nx.Graph()
        names_vars = correlation_result.names_variables
        set_anchors = set(selection_result.indices_selected)

        # Mapping variable index to cluster id
        dict_var_to_cluster: ty.Dict[int, int] = {
            m.id_variable: m.id_cluster for m in clustering_result.list_memberships
        }

        # Add nodes
        for idx_var, name_var in enumerate(names_vars):
            graph_network.add_node(
                idx_var,
                name=name_var,
                is_anchor=(idx_var in set_anchors),
                cluster=dict_var_to_cluster.get(idx_var, 0)
            )
        # end for idx_var

        # Add edges
        for edge in correlation_result.list_edges:
            graph_network.add_edge(
                edge.id_variable_1,
                edge.id_variable_2,
                weight=edge.correlation_score
            )
        # end for edge

        fig, ax = plt.subplots(figsize=(14, 11), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")

        # Spring layout
        pos = nx.spring_layout(graph_network, k=0.35, iterations=60, seed=42)

        # Colormap for clusters
        unique_clusters = sorted(list(set(dict_var_to_cluster.values())))
        cmap = matplotlib.colormaps.get_cmap("tab10")

        # Draw edges with opacity based on weight
        for u, v, data in graph_network.edges(data=True):
            weight = data.get("weight", 0.3)
            nx.draw_networkx_edges(
                graph_network,
                pos,
                edgelist=[(u, v)],
                width=1.0 + 2.0 * weight,
                alpha=min(0.7, max(0.15, float(weight))),
                edge_color="#64748b",
                ax=ax,
            )
        # end for

        # Draw nodes
        for cluster_id in unique_clusters:
            nodes_in_cluster = [
                n for n, d in graph_network.nodes(data=True) if d.get("cluster") == cluster_id
            ]
            if not nodes_in_cluster:
                continue
            # end if

            color = cmap(cluster_id % 10)

            # Regular nodes
            regular_nodes = [n for n in nodes_in_cluster if not graph_network.nodes[n]["is_anchor"]]
            if regular_nodes:
                nx.draw_networkx_nodes(
                    graph_network,
                    pos,
                    nodelist=regular_nodes,
                    node_size=self.size_regular_node,
                    node_color=[color],
                    alpha=0.8,
                    ax=ax,
                )
            # end if

            # Anchor nodes (larger, highlighted edge)
            anchor_nodes = [n for n in nodes_in_cluster if graph_network.nodes[n]["is_anchor"]]
            if anchor_nodes:
                nx.draw_networkx_nodes(
                    graph_network,
                    pos,
                    nodelist=anchor_nodes,
                    node_size=self.size_anchor_node,
                    node_color=[color],
                    edgecolors="#f8fafc",
                    linewidths=2.5,
                    alpha=1.0,
                    ax=ax,
                )
            # end if
        # end for cluster_id

        # Labels for anchor nodes and high-degree nodes
        labels = {}
        for n, d in graph_network.nodes(data=True):
            if d.get("is_anchor"):
                labels[n] = f"★ {d['name']}"
            elif graph_network.degree(n) >= 4:
                labels[n] = d["name"]
            # end if
        # end for

        nx.draw_networkx_labels(
            graph_network,
            pos,
            labels=labels,
            font_size=8,
            font_color="#f8fafc",
            font_family="sans-serif",
            ax=ax,
        )

        ax.set_title(
            "Constellation Network: Global Variable Landscape & MMD Anchors",
            fontsize=15,
            color="#f8fafc",
            pad=18,
            fontweight="bold"
        )
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(path_output_image, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        return os.path.abspath(path_output_image)
        # end def plot_constellation_graph
# end class ConstellationGraphPlotter
