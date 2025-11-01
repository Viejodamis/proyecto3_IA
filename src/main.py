"""CLI runner for the simple Bayesian network loader."""
from pathlib import Path
from src.bayesnet import build_bayesnet, show_graph
import argparse


def main():
    p = argparse.ArgumentParser(description='Load Bayesian network from CSV and show/save graph')
    p.add_argument('--data', '-d', default=str(Path(__file__).resolve().parents[1] / 'data'),
                   help='Path to folder containing edges.csv and cpt_*.csv')
    p.add_argument('--out', '-o', default=None, help='Path to save graph image (png). If omitted, saves to data/graph.png')
    args = p.parse_args()

    data_folder = Path(args.data)
    edges = data_folder / 'edges.csv'
    out = Path(args.out) if args.out else (data_folder / 'graph.png')

    G = build_bayesnet(edges, data_folder)
    show_graph(G, save_path=out)


if __name__ == '__main__':
    main()
