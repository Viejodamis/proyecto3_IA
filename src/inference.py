"""Inferencia por enumeración en redes bayesianas con trazado detallado.

Este módulo implementa el algoritmo de enumeración para inferencia exacta en
redes bayesianas discretas, con un registro detallado del proceso de cómputo.
"""
import itertools
from collections import defaultdict
from pathlib import Path
import pandas as pd
import networkx as nx


class RastreadorInferencia:
    """Rastrea y registra los pasos de cómputo de la inferencia."""
    def __init__(self, archivo_log=None):
        self.pasos = []
        self.archivo_log = Path(archivo_log) if archivo_log else None
        if self.archivo_log:
            # Iniciar log nuevo
            self.archivo_log.write_text('')
    
    def agregar_paso(self, msg):
        """Agrega un paso de cómputo y opcionalmente lo escribe al archivo."""
        print(msg)  # Siempre imprimir a consola
        self.pasos.append(msg)
        if self.archivo_log:
            with open(self.archivo_log, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')


def enumerar_todo(variables, evidencia, G, vars_red, rastreador):
    """Retorna la distribución sobre la variable de consulta por enumeración.
    
    Args:
        variables: List[str], variables a enumerar (en orden topológico)
        evidencia: dict, variable -> asignación de valores
        G: networkx.DiGraph con CPTs almacenadas en atributos de nodos
        vars_red: dict que mapea cada variable a sus valores posibles
        rastreador: RastreadorInferencia para registrar pasos
    
    Returns:
        float: probabilidad de la evidencia
    """
    if not variables:
        return 1.0
    
    Y, resto = variables[0], variables[1:]
    rastreador.agregar_paso(f"\nEnumerando sobre {Y}")
    rastreador.agregar_paso(f"  Evidencia actual: {evidencia}")
    
    if Y in evidencia:
        # Variable ya tiene valor en evidencia
        py = obtener_probabilidad(Y, evidencia, G)
        rastreador.agregar_paso(f"  {Y} en evidencia, P({Y}={evidencia[Y]}|padres)={py:.4f}")
        resultado = py * enumerar_todo(resto, evidencia, G, vars_red, rastreador)
        rastreador.agregar_paso(f"  Retornando {resultado:.4f}")
        return resultado
    
    # Sumar sobre valores posibles de Y
    total = 0
    rastreador.agregar_paso(f"  Sumando sobre valores de {Y}: {vars_red[Y]}")
    for y in vars_red[Y]:
        evidencia[Y] = y
        py = obtener_probabilidad(Y, evidencia, G)
        rastreador.agregar_paso(f"    P({Y}={y}|padres)={py:.4f}")
        sub = py * enumerar_todo(resto, evidencia, G, vars_red, rastreador)
        rastreador.agregar_paso(f"    Término para {Y}={y}: {sub:.4f}")
        total += sub
    evidencia.pop(Y)  # Eliminar de evidencia antes de retornar
    rastreador.agregar_paso(f"  Suma para {Y}: {total:.4f}")
    return total


def obtener_probabilidad(var, evidencia, G):
    """Retorna la probabilidad de var=val dados los valores de los padres en evidencia.
    
    La CPT de cada nodo debe estar almacenada en G.nodes[var]['cpt'] como DataFrame
    con columnas para valores de padres (si hay) y 'value', 'prob'.
    """
    cpt = G.nodes[var]['cpt']
    # Obtener padres y sus valores de evidencia
    padres = list(G.predecessors(var))
    if not padres:
        # Sin padres - solo buscar probabilidad
        return float(cpt[cpt['value'] == evidencia[var]]['prob'].iloc[0])
    
    # Buscar valores de padres en CPT
    consulta = {p: evidencia[p] for p in padres}
    consulta['value'] = evidencia[var]
    # Usar indexación booleana de pandas para encontrar fila
    coincidencias = cpt
    for col, val in consulta.items():
        coincidencias = coincidencias[coincidencias[col] == val]
    return float(coincidencias['prob'].iloc[0])


def consulta_enumeracion(X, evidencia, G, vars_red=None, archivo_log=None):
    """Retorna distribución sobre X por enumeración dada la evidencia.
    
    Args:
        X: str, variable de consulta
        evidencia: dict con mapeo de variables a valores
        G: networkx.DiGraph con CPTs almacenadas en atributos de nodos
        vars_red: dict opcional que mapea variables a sus valores posibles
                (por defecto {True, False} para todas las variables)
        archivo_log: ruta opcional para escribir traza de cómputo
    
    Returns:
        Distribución sobre X como dict que mapea valores a probabilidades
    """
    if vars_red is None:
        # Por defecto variables binarias
        vars_red = {var: {True, False} for var in G.nodes}
    
    rastreador = RastreadorInferencia(archivo_log)
    rastreador.agregar_paso(f"\nCalculando P({X}|{evidencia})")
    
    # Obtener variables en orden topológico (asegura orden correcto de enumeración)
    variables = list(nx.topological_sort(G))
    rastreador.agregar_paso(f"Variables en orden topológico: {variables}")
    
    # Calcular distribución normalizando sobre valores de variable de consulta
    Q = defaultdict(float)
    for x in vars_red[X]:
        evidencia[X] = x
        rastreador.agregar_paso(f"\nCalculando P({X}={x}, e)")
        Q[x] = enumerar_todo(variables, evidencia, G, vars_red, rastreador)
        rastreador.agregar_paso(f"P({X}={x}, e) = {Q[x]:.4f}")
    evidencia.pop(X)
    
    # Normalizar
    total = sum(Q.values())
    for x in Q:
        Q[x] /= total
        rastreador.agregar_paso(f"P({X}={x}|e) = {Q[x]:.4f}")
    
    return dict(Q)  # Convertir defaultdict a dict normal


if __name__ == '__main__':
    # Pequeña prueba/demo
    from bayesnet import build_bayesnet
    
    ruta_aristas = Path(__file__).resolve().parents[1] / 'data' / 'edges.csv'
    carpeta_cpt = ruta_aristas.parent
    G = build_bayesnet(ruta_aristas, carpeta_cpt)
    
    # P(Lluvia | CespedMojado=true)
    evidencia = {'GrassWet': True}
    dist = consulta_enumeracion('Rain', evidencia, G, archivo_log='traza.txt')
    print(f"\nP(Lluvia|CespedMojado=true) = {dist}")