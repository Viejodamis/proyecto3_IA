"""Runner CLI para carga de red bayesiana e inferencia."""
from pathlib import Path
import argparse

from src.bayesnet import construir_red_bayesiana, mostrar_grafo
from src.inference import consulta_enumeracion


def imprimir_distribucion(dist):
    """Imprime bonito una distribución dict."""
    print("{")
    for val, prob in sorted(dist.items()):
        print(f"    {val}: {prob:.4f}")
    print("}")


def main():
    p = argparse.ArgumentParser(description='Cargar red bayesiana y ejecutar inferencia')
    p.add_argument('--data', '-d', default=str(Path(__file__).resolve().parents[1] / 'data'),
                   help='Ruta a carpeta con edges.csv y cpt_*.csv')
    p.add_argument('--out', '-o', default=None, 
                   help='Ruta para guardar imagen del grafo (png). Si se omite, guarda en data/grafo.png')
    args = p.parse_args()

    carpeta_data = Path(args.data)
    aristas = carpeta_data / 'edges.csv'
    salida = Path(args.out) if args.out else (carpeta_data / 'grafo.png')

    print("Construyendo red bayesiana...")
    G = construir_red_bayesiana(aristas, carpeta_data)
    mostrar_grafo(G, ruta_guardado=salida)

    print("\nDemostrando inferencia por enumeración...")
    print("\nEjemplo 1: P(Lluvia|CespedMojado=true)")
    print("Calculando probabilidad posterior de que llovió, dado que el césped está mojado")
    dist = consulta_enumeracion('Rain', {'GrassWet': True}, G, 
                             archivo_log=carpeta_data / 'traza_lluvia_dado_mojado.txt')
    print("\nResultado:")
    imprimir_distribucion(dist)

    print("\nEjemplo 2: P(Aspersor|CespedMojado=true, Lluvia=false)")
    print("Calculando probabilidad de que el aspersor estaba encendido, dado césped mojado sin lluvia")
    dist = consulta_enumeracion('Sprinkler', {'GrassWet': True, 'Rain': False}, G,
                             archivo_log=carpeta_data / 'traza_aspersor_dado_mojado_sinlluvia.txt')
    print("\nResultado:")
    imprimir_distribucion(dist)

    print("\nTrazas detalladas de cómputo guardadas en:")
    print(f"1. {carpeta_data}/traza_lluvia_dado_mojado.txt")
    print(f"2. {carpeta_data}/traza_aspersor_dado_mojado_sinlluvia.txt")


if __name__ == '__main__':
    main()