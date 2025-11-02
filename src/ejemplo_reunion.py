"""Script para probar el ejemplo de la reunión."""
from pathlib import Path
from src.bayesnet import construir_red_bayesiana, mostrar_grafo
from src.inference import consulta_enumeracion

def main():
    # Cargar red bayesiana para el ejemplo de la reunión
    carpeta_data = Path(__file__).resolve().parents[1] / 'data' / 'reunion'
    aristas = carpeta_data / 'edges.csv'

    print("Construyendo red bayesiana del ejemplo de la reunión...")
    G = construir_red_bayesiana(aristas, carpeta_data)
    mostrar_grafo(G, ruta_guardado=carpeta_data / 'grafo_reunion.png')

    # Definir valores posibles para cada variable
    valores_variables = {
        'Rain': {'none', 'light', 'heavy'},
        'Maintenance': {'yes', 'no'},
        'Train': {'on_time', 'delayed'},
        'Appointment': {'yes', 'no'}
    }

    # Consulta: P(Appointment | Rain=light ∧ Maintenance=no)
    print("\nEjemplo: P(Appointment | Rain=light ∧ Maintenance=no)")
    print("Calculando la probabilidad de llegar a la reunión dado que hay lluvia ligera y no hay mantenimiento")
    
    evidencia = {'Rain': 'light', 'Maintenance': 'no'}
    dist = consulta_enumeracion('Appointment', evidencia, G, 
                            vars_red=valores_variables,
                            archivo_log=carpeta_data / 'traza_reunion.txt')
    
    print("\nResultado P(Appointment | Rain=light ∧ Maintenance=no):")
    print("{")
    for val, prob in sorted(dist.items()):
        print(f"    {val}: {prob:.4f}")
    print("}")

    # Mostrar ruta de traza
    print("\nTraza detallada del cómputo guardada en:")
    print(f"{carpeta_data}/traza_reunion.txt")

if __name__ == '__main__':
    main()