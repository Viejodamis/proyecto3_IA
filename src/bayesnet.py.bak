"""Simple Bayesian network loader from CSV files.

Files expected:
- edges CSV: columns parent,child
- CPT files: named cpt_<Node>.csv. For nodes with parents include parent column(s) matching parent names.

This module builds a networkx DiGraph with CPTs stored as node attribute 'cpt' (a pandas.DataFrame).
"""
from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def read_edges(edges_csv_path):
    """Read edges CSV and return list of (parent, child) tuples."""
    df = pd.read_csv(edges_csv_path)
    if 'parent' not in df.columns or 'child' not in df.columns:
        raise ValueError("edges CSV must have 'parent' and 'child' columns")
    return list(df[['parent', 'child']].itertuples(index=False, name=None))


def read_cpts(cpt_folder):
    """Read all cpt_*.csv files in folder and return dict node -> DataFrame."""
    folder = Path(cpt_folder)
    cpts = {}
    for p in sorted(folder.glob('cpt_*.csv')):
        node = p.stem[len('cpt_'):]
        df = pd.read_csv(p)
        cpts[node] = df
    return cpts


def build_bayesnet(edges_csv_path, cpt_folder):
    """Build and return a networkx.DiGraph with CPTs attached.

    Node attribute 'cpt' holds the pandas.DataFrame for that node's CPT (if provided).
    """
    G = nx.DiGraph()
    edges = read_edges(edges_csv_path)
    # add edges and nodes
    G.add_edges_from(edges)

    # ensure all nodes exist even if isolated
    nodes = set()
    for p, c in edges:
        nodes.add(p)
        nodes.add(c)
    for n in nodes:
        if n not in G:
            G.add_node(n)

    # attach CPTs if available
    cpts = read_cpts(cpt_folder)
    for node, df in cpts.items():
        if node not in G:
            G.add_node(node)
        G.nodes[node]['cpt'] = df

    return G


def show_graph(G, save_path=None, figsize=(8,6)):
    """Render graph with node labels and save to file (if save_path).

    Also prints a brief CPT summary for each node to stdout.
    """
    print("Nodes:")
    for n in G.nodes:
        print(f" - {n}")
    print("\nEdges:")
    for u, v in G.edges:
        print(f" - {u} -> {v}")

    print("\nCPT summaries:")
    for n in G.nodes:
        cpt = G.nodes[n].get('cpt')
        if cpt is None:
            print(f" - {n}: (no CPT file found)")
        else:
            # print first few rows
            print(f" - {n}: \n{cpt.head().to_string(index=False)}\n")

    # draw graph
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='#a6cee3', font_size=10, arrowsize=20)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Graph image saved to: {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    # simple quick demo when run directly (looks for ../data by default)
    base = Path(__file__).resolve().parents[1]
    edges = base / 'data' / 'edges.csv'
    cpt_folder = base / 'data'
    G = build_bayesnet(edges, cpt_folder)
    show_graph(G, save_path=base / 'data' / 'graph.png')
