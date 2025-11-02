"""Inference by enumeration in Bayesian networks with detailed tracing.

This module implements the enumeration algorithm for exact inference in discrete
Bayesian networks, with detailed tracing of the computation process.
"""
import itertools
from collections import defaultdict
from pathlib import Path
import pandas as pd
import networkx as nx


class InferenceTracer:
    """Track and log inference computation steps."""
    def __init__(self, log_file=None):
        self.steps = []
        self.log_file = Path(log_file) if log_file else None
        if self.log_file:
            # Start fresh log
            self.log_file.write_text('')
    
    def add_step(self, msg):
        """Add a computation step and optionally write to log file."""
        print(msg)  # Always print to console
        self.steps.append(msg)
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')


def enumerate_all(variables, evidence, G, bn_vars, tracer):
    """Return distribution over query variable by enumeration.
    
    Args:
        variables: List[str], variables to enumerate over (in topological order)
        evidence: dict, variable -> value assignments
        G: networkx.DiGraph with CPTs stored in node attributes
        bn_vars: dict mapping each variable to its possible values
        tracer: InferenceTracer to log computation steps
    
    Returns:
        float: probability of evidence
    """
    if not variables:
        return 1.0
    
    Y, rest = variables[0], variables[1:]
    tracer.add_step(f"\nEnumerating over {Y}")
    tracer.add_step(f"  Current evidence: {evidence}")
    
    if Y in evidence:
        # Variable already has value in evidence
        py = probability(Y, evidence, G)
        tracer.add_step(f"  {Y} in evidence, P({Y}={evidence[Y]}|parents)={py:.4f}")
        result = py * enumerate_all(rest, evidence, G, bn_vars, tracer)
        tracer.add_step(f"  Returning {result:.4f}")
        return result
    
    # Sum over possible values of Y
    total = 0
    tracer.add_step(f"  Summing over values of {Y}: {bn_vars[Y]}")
    for y in bn_vars[Y]:
        evidence[Y] = y
        py = probability(Y, evidence, G)
        tracer.add_step(f"    P({Y}={y}|parents)={py:.4f}")
        sub = py * enumerate_all(rest, evidence, G, bn_vars, tracer)
        tracer.add_step(f"    Term for {Y}={y}: {sub:.4f}")
        total += sub
    evidence.pop(Y)  # Remove from evidence before returning
    tracer.add_step(f"  Sum for {Y}: {total:.4f}")
    return total


def probability(var, evidence, G):
    """Return probability of var=val given parents values in evidence.
    
    The CPT for each node should be stored in G.nodes[var]['cpt'] as a pandas DataFrame
    with columns for parent values (if any) and 'value', 'prob'.
    """
    cpt = G.nodes[var]['cpt']
    # Get parents and their values from evidence
    parents = list(G.predecessors(var))
    if not parents:
        # No parents - just look up probability
        return float(cpt[cpt['value'] == evidence[var]]['prob'].iloc[0])
    
    # Match parent values in CPT
    query = {p: evidence[p] for p in parents}
    query['value'] = evidence[var]
    # Use pandas boolean indexing to find matching row
    matches = cpt
    for col, val in query.items():
        matches = matches[matches[col] == val]
    return float(matches['prob'].iloc[0])


def enumeration_ask(X, evidence, G, bn_vars=None, log_file=None):
    """Return distribution over X by enumeration given evidence.
    
    Args:
        X: str, query variable
        evidence: dict mapping variables to values
        G: networkx.DiGraph with CPTs stored in node attributes
        bn_vars: optional dict mapping variables to their possible values
               (defaults to {True, False} for all variables)
        log_file: optional path to write computation trace
    
    Returns:
        Distribution over X as dict mapping values to probabilities
    """
    if bn_vars is None:
        # Default to binary variables
        bn_vars = {var: {True, False} for var in G.nodes}
    
    tracer = InferenceTracer(log_file)
    tracer.add_step(f"\nComputing P({X}|{evidence})")
    
    # Get all variables in topological order (ensures correct enumeration order)
    variables = list(nx.topological_sort(G))
    tracer.add_step(f"Variables in topological order: {variables}")
    
    # Compute distribution by normalizing across query variable values
    Q = defaultdict(float)
    for x in bn_vars[X]:
        evidence[X] = x
        tracer.add_step(f"\nComputing P({X}={x}, e)")
        Q[x] = enumerate_all(variables, evidence, G, bn_vars, tracer)
        tracer.add_step(f"P({X}={x}, e) = {Q[x]:.4f}")
    evidence.pop(X)
    
    # Normalize
    total = sum(Q.values())
    for x in Q:
        Q[x] /= total
        tracer.add_step(f"P({X}={x}|e) = {Q[x]:.4f}")
    
    return dict(Q)  # Convert defaultdict to regular dict


if __name__ == '__main__':
    # Small test/demo
    from bayesnet import build_bayesnet
    
    edges = Path(__file__).resolve().parents[1] / 'data' / 'edges.csv'
    cpt_folder = edges.parent
    G = build_bayesnet(edges, cpt_folder)
    
    # P(Rain | GrassWet=true)
    evidence = {'GrassWet': True}
    dist = enumeration_ask('Rain', evidence, G, log_file='trace.txt')
    print(f"\nP(Rain|GrassWet=true) = {dist}")