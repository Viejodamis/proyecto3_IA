"""Cargador simple de redes bayesianas desde archivos CSV.

Archivos esperados:
- CSV de aristas: columnas parent,child
- archivos CPT: nombrados cpt_<Nodo>.csv. Para nodos con padres incluir columna(s)
  de padres que coincidan con los nombres de los padres.

Este módulo construye un networkx DiGraph con CPTs almacenadas como atributo 'cpt' del nodo
(un pandas.DataFrame).
"""
from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def leer_aristas(ruta_csv_aristas):
    """Lee CSV de aristas y retorna lista de tuplas (padre, hijo)."""
    df = pd.read_csv(ruta_csv_aristas)
    if 'parent' not in df.columns or 'child' not in df.columns:
        raise ValueError("El CSV de aristas debe tener columnas 'parent' y 'child'")
    return list(df[['parent', 'child']].itertuples(index=False, name=None))


def leer_cpts(carpeta_cpt):
    """Lee todos los archivos cpt_*.csv en la carpeta y retorna dict nodo -> DataFrame."""
    carpeta = Path(carpeta_cpt)
    cpts = {}
    for p in sorted(carpeta.glob('cpt_*.csv')):
        nodo = p.stem[len('cpt_'):]
        df = pd.read_csv(p)
        cpts[nodo] = df
    return cpts


def construir_red_bayesiana(ruta_csv_aristas, carpeta_cpt):
    """Construye y retorna un networkx.DiGraph con CPTs adjuntas.

    El atributo 'cpt' del nodo contiene el pandas.DataFrame para la CPT de ese nodo (si existe).
    """
    G = nx.DiGraph()
    aristas = leer_aristas(ruta_csv_aristas)
    # agregar aristas y nodos
    G.add_edges_from(aristas)

    # asegurar que todos los nodos existan incluso si están aislados
    nodos = set()
    for p, c in aristas:
        nodos.add(p)
        nodos.add(c)
    for n in nodos:
        if n not in G:
            G.add_node(n)

    # adjuntar CPTs si están disponibles
    cpts = leer_cpts(carpeta_cpt)
    for nodo, df in cpts.items():
        if nodo not in G:
            G.add_node(nodo)
        G.nodes[nodo]['cpt'] = df

    return G


def mostrar_grafo(G, ruta_guardado=None, tamano_fig=(8,6)):
    """Renderiza grafo con etiquetas de nodos y guarda a archivo (si ruta_guardado).

    También imprime un breve resumen de CPTs para cada nodo en stdout.
    """
    print("Nodos:")
    for n in G.nodes:
        print(f" - {n}")
    print("\nAristas:")
    for u, v in G.edges:
        print(f" - {u} -> {v}")

    print("\nResumen de CPTs:")
    for n in G.nodes:
        cpt = G.nodes[n].get('cpt')
        if cpt is None:
            print(f" - {n}: (no se encontró archivo CPT)")
        else:
            # imprimir primeras filas
            print(f" - {n}: \n{cpt.head().to_string(index=False)}\n")

    # dibujar grafo
    plt.figure(figsize=tamano_fig)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='#a6cee3', 
            font_size=10, arrowsize=20)
    if ruta_guardado:
        plt.savefig(ruta_guardado, bbox_inches='tight')
        print(f"Imagen del grafo guardada en: {ruta_guardado}")
    else:
        plt.show()


if __name__ == '__main__':
    # pequeña demo rápida cuando se ejecuta directamente (busca ../data por defecto)
    base = Path(__file__).resolve().parents[1]
    aristas = base / 'data' / 'edges.csv'
    carpeta_cpt = base / 'data'
    G = construir_red_bayesiana(aristas, carpeta_cpt)
    mostrar_grafo(G, ruta_guardado=base / 'data' / 'grafo.png')