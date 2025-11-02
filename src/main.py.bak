"""CLI runner for Bayesian network loading and inference."""
from pathlib import Path
import argparse

from src.bayesnet import build_bayesnet, show_graph
from src.inference import enumeration_ask


def print_distribution(dist):
    """Pretty print a distribution dict."""
    print("{")
    for val, prob in sorted(dist.items()):
        print(f"    {val}: {prob:.4f}")
    print("}")


def main():
    p = argparse.ArgumentParser(description='Load Bayesian network and run inference')
    p.add_argument('--data', '-d', default=str(Path(__file__).resolve().parents[1] / 'data'),
                   help='Path to folder containing edges.csv and cpt_*.csv')
    p.add_argument('--out', '-o', default=None, 
                   help='Path to save graph image (png). If omitted, saves to data/graph.png')
    args = p.parse_args()

    data_folder = Path(args.data)
    edges = data_folder / 'edges.csv'
    out = Path(args.out) if args.out else (data_folder / 'graph.png')

    print("Building Bayesian network...")
    G = build_bayesnet(edges, data_folder)
    show_graph(G, save_path=out)

    print("\nDemonstrating inference by enumeration...")
    print("\nExample 1: P(Rain|GrassWet=true)")
    print("Computing posterior probability that it rained, given that grass is wet")
    dist = enumeration_ask('Rain', {'GrassWet': True}, G, 
                          log_file=data_folder / 'trace_rain_given_wet.txt')
    print("\nResult:")
    print_distribution(dist)

    print("\nExample 2: P(Sprinkler|GrassWet=true, Rain=false)")
    print("Computing probability sprinkler was on, given grass is wet but it didn't rain")
    dist = enumeration_ask('Sprinkler', {'GrassWet': True, 'Rain': False}, G,
                          log_file=data_folder / 'trace_sprinkler_given_wet_norain.txt')
    print("\nResult:")
    print_distribution(dist)

    print("\nDetailed computation traces have been saved to:")
    print(f"1. {data_folder}/trace_rain_given_wet.txt")
    print(f"2. {data_folder}/trace_sprinkler_given_wet_norain.txt")


if __name__ == '__main__':
    main()
