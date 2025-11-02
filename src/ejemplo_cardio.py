"""Ejemplo de diagnóstico cardíaco usando red bayesiana.

Este ejemplo modela un sistema simple de diagnóstico cardíaco considerando:
- Factores de riesgo: Edad, Obesidad, Sedentarismo
- Signos: Presión Alta
- Síntomas: Dolor en el Pecho, Fatiga
- Diagnóstico: Condición Cardíaca

La red permite calcular probabilidades de diagnóstico basadas en:
- Factores de riesgo observados
- Síntomas reportados
- Signos medidos
"""
from pathlib import Path
from src.bayesnet import construir_red_bayesiana, mostrar_grafo
from src.inference import consulta_enumeracion


def imprimir_distribucion(dist):
    """Imprime bonito una distribución dict."""
    print("{")
    for val, prob in sorted(dist.items()):
        print(f"    {val}: {prob:.4f}")
    print("}")


def main():
    # Cargar red bayesiana para el ejemplo de diagnóstico cardíaco
    carpeta_data = Path(__file__).resolve().parents[1] / 'data' / 'cardio'
    aristas = carpeta_data / 'edges.csv'

    # Definir valores posibles para cada variable
    valores_variables = {
        'Edad': {'joven', 'adulto', 'mayor'},
        'Obesidad': {'si', 'no'},
        'Sedentarismo': {'si', 'no'},
        'PresionAlta': {'si', 'no'},
        'DolorPecho': {'si', 'no'},
        'Fatiga': {'si', 'no'},
        'DiagnosticoCardio': {'si', 'no'}
    }

    print("Construyendo red bayesiana del sistema de diagnóstico cardíaco...")
    G = construir_red_bayesiana(aristas, carpeta_data)
    mostrar_grafo(G, ruta_guardado=carpeta_data / 'grafo_cardio.png')

    print("\nEjemplo 1: P(DiagnosticoCardio | Edad=mayor, PresionAlta=si)")
    print("Calculando probabilidad de condición cardíaca en persona mayor con presión alta")
    evidencia = {'Edad': 'mayor', 'PresionAlta': 'si'}
    dist = consulta_enumeracion('DiagnosticoCardio', evidencia, G,
                               vars_red=valores_variables,
                               archivo_log=carpeta_data / 'traza_mayor_presion.txt')
    print("\nResultado:")
    imprimir_distribucion(dist)

    print("\nEjemplo 2: P(DiagnosticoCardio | Edad=adulto, Obesidad=si, Sedentarismo=si, DolorPecho=si)")
    print("Calculando probabilidad de condición cardíaca en adulto obeso y sedentario con dolor en el pecho")
    evidencia = {
        'Edad': 'adulto',
        'Obesidad': 'si',
        'Sedentarismo': 'si',
        'DolorPecho': 'si'
    }
    dist = consulta_enumeracion('DiagnosticoCardio', evidencia, G,
                               vars_red=valores_variables,
                               archivo_log=carpeta_data / 'traza_adulto_riesgo.txt')
    print("\nResultado:")
    imprimir_distribucion(dist)

    print("\nEjemplo 3: P(DiagnosticoCardio | Edad=joven, Obesidad=no, Sedentarismo=no, Fatiga=si)")
    print("Calculando probabilidad de condición cardíaca en joven saludable pero con fatiga")
    evidencia = {
        'Edad': 'joven',
        'Obesidad': 'no',
        'Sedentarismo': 'no',
        'Fatiga': 'si'
    }
    dist = consulta_enumeracion('DiagnosticoCardio', evidencia, G,
                               vars_red=valores_variables,
                               archivo_log=carpeta_data / 'traza_joven_fatiga.txt')
    print("\nResultado:")
    imprimir_distribucion(dist)

    # Mostrar rutas de trazas
    print("\nTrazas detalladas guardadas en:")
    print(f"1. {carpeta_data}/traza_mayor_presion.txt")
    print(f"2. {carpeta_data}/traza_adulto_riesgo.txt")
    print(f"3. {carpeta_data}/traza_joven_fatiga.txt")


if __name__ == '__main__':
    main()