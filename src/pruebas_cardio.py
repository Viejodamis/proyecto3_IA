"""Casos de prueba para el diagnóstico cardíaco.

Casos de prueba con variables ocultas y sus respectivos cálculos manuales para validación.
"""
from pathlib import Path
from src.bayesnet import construir_red_bayesiana
from src.inference import consulta_enumeracion


def test_persona_mayor_presion_alta():
    """Caso 1: Persona mayor con presión alta - ¿probabilidad de diagnóstico cardíaco?

    Variables evidencia: Edad=mayor, PresionAlta=si
    Variables ocultas: Obesidad, Sedentarismo, DolorPecho, Fatiga

    Cálculo esperado (a mano):
    - Para DiagnosticoCardio=si:
        
        P(DC=si|E=mayor,PA=si) = 
        ∑(O,S,DP,F) P(E=mayor)P(O)P(S)P(PA=si|E=mayor,O,S)P(DP)P(F|O,S)P(DC=si|E=mayor,PA=si,DP,F) 
        / P(E=mayor,PA=si)
        
        = 0.6331 (63.31%)

    - Explicación paso a paso:
        1. P(E=mayor) = 0.2 
        2. Para cada combinación de variables ocultas:
            - P(O=si) = 0.4, P(O=no) = 0.6
            - P(S=si) = 0.7, P(S=no) = 0.3
            - P(DP=si) = 0.3, P(DP=no) = 0.7 
            - P(F) depende de O y S según CPT
            - P(DC) depende de E, PA, DP y F según CPT
    """
    carpeta_data = Path(__file__).resolve().parents[1] / 'data' / 'cardio'
    aristas = carpeta_data / 'edges.csv'

    valores_variables = {
        'Edad': {'joven', 'adulto', 'mayor'},
        'Obesidad': {'si', 'no'},
        'Sedentarismo': {'si', 'no'}, 
        'PresionAlta': {'si', 'no'},
        'DolorPecho': {'si', 'no'},
        'Fatiga': {'si', 'no'},
        'DiagnosticoCardio': {'si', 'no'}
    }

    G = construir_red_bayesiana(aristas, carpeta_data)
    evidencia = {'Edad': 'mayor', 'PresionAlta': 'si'}
    dist = consulta_enumeracion('DiagnosticoCardio', evidencia, G,
                               vars_red=valores_variables)
    
    # Validar resultado
    valor_esperado = 0.6331  # 63.31%
    valor_obtenido = dist['si']
    print(f"\nTest 1: P(DiagnosticoCardio | Edad=mayor, PresionAlta=si)")
    print(f"Valor esperado: {valor_esperado:.4f}")
    print(f"Valor obtenido: {valor_obtenido:.4f}")
    print(f"Error absoluto: {abs(valor_esperado - valor_obtenido):.4f}")


def test_persona_joven_sedentaria():
    """Caso 2: Persona joven sedentaria - ¿probabilidad de diagnóstico cardíaco?

    Variables evidencia: Edad=joven, Sedentarismo=si
    Variables ocultas: Obesidad, PresionAlta, DolorPecho, Fatiga

    Cálculo esperado (a mano):
    - Para DiagnosticoCardio=si:
        
        P(DC=si|E=joven,S=si) = 
        ∑(O,PA,DP,F) P(E=joven)P(O)P(S=si)P(PA|E=joven,O,S=si)P(DP)P(F|O,S=si)P(DC=si|E=joven,PA,DP,F)
        / P(E=joven,S=si)
        
        = 0.2214 (22.14%)

    - Explicación paso a paso:
        1. P(E=joven) = 0.3
        2. P(S=si) = 0.7
        3. Para cada combinación de variables ocultas:
            - P(O=si) = 0.4, P(O=no) = 0.6
            - P(DP=si) = 0.3, P(DP=no) = 0.7
            - P(PA) depende de E, O y S según CPT
            - P(F) depende de O y S según CPT
            - P(DC) depende de E, PA, DP y F según CPT
    """
    carpeta_data = Path(__file__).resolve().parents[1] / 'data' / 'cardio'
    aristas = carpeta_data / 'edges.csv'

    valores_variables = {
        'Edad': {'joven', 'adulto', 'mayor'},
        'Obesidad': {'si', 'no'},
        'Sedentarismo': {'si', 'no'}, 
        'PresionAlta': {'si', 'no'},
        'DolorPecho': {'si', 'no'},
        'Fatiga': {'si', 'no'},
        'DiagnosticoCardio': {'si', 'no'}
    }

    G = construir_red_bayesiana(aristas, carpeta_data)
    evidencia = {'Edad': 'joven', 'Sedentarismo': 'si'}
    dist = consulta_enumeracion('DiagnosticoCardio', evidencia, G,
                               vars_red=valores_variables)
    
    # Validar resultado
    valor_esperado = 0.2214  # 22.14%
    valor_obtenido = dist['si']
    print(f"\nTest 2: P(DiagnosticoCardio | Edad=joven, Sedentarismo=si)")
    print(f"Valor esperado: {valor_esperado:.4f}")
    print(f"Valor obtenido: {valor_obtenido:.4f}")
    print(f"Error absoluto: {abs(valor_esperado - valor_obtenido):.4f}")


def test_persona_adulta_sintomas():
    """Caso 3: Persona adulta con síntomas - ¿probabilidad de diagnóstico cardíaco?

    Variables evidencia: Edad=adulto, DolorPecho=si, Fatiga=si
    Variables ocultas: Obesidad, Sedentarismo, PresionAlta

    Cálculo esperado (a mano):
    - Para DiagnosticoCardio=si:
        
        P(DC=si|E=adulto,DP=si,F=si) = 
        ∑(O,S,PA) P(E=adulto)P(O)P(S)P(PA|E=adulto,O,S)P(DP=si)P(F=si|O,S)P(DC=si|E=adulto,PA,DP=si,F=si)
        / P(E=adulto,DP=si,F=si)
        
        = 0.5812 (58.12%)

    - Explicación paso a paso:
        1. P(E=adulto) = 0.5
        2. P(DP=si) = 0.3
        3. Para cada combinación de variables ocultas:
            - P(O=si) = 0.4, P(O=no) = 0.6
            - P(S=si) = 0.7, P(S=no) = 0.3
            - P(PA) depende de E, O y S según CPT
            - P(F=si) depende de O y S según CPT
            - P(DC) depende de E, PA, DP y F según CPT
    """
    carpeta_data = Path(__file__).resolve().parents[1] / 'data' / 'cardio'
    aristas = carpeta_data / 'edges.csv'

    valores_variables = {
        'Edad': {'joven', 'adulto', 'mayor'},
        'Obesidad': {'si', 'no'},
        'Sedentarismo': {'si', 'no'}, 
        'PresionAlta': {'si', 'no'},
        'DolorPecho': {'si', 'no'},
        'Fatiga': {'si', 'no'},
        'DiagnosticoCardio': {'si', 'no'}
    }

    G = construir_red_bayesiana(aristas, carpeta_data)
    evidencia = {'Edad': 'adulto', 'DolorPecho': 'si', 'Fatiga': 'si'}
    dist = consulta_enumeracion('DiagnosticoCardio', evidencia, G,
                               vars_red=valores_variables)
    
    # Validar resultado
    valor_esperado = 0.5812  # 58.12%
    valor_obtenido = dist['si']
    print(f"\nTest 3: P(DiagnosticoCardio | Edad=adulto, DolorPecho=si, Fatiga=si)")
    print(f"Valor esperado: {valor_esperado:.4f}")
    print(f"Valor obtenido: {valor_obtenido:.4f}")
    print(f"Error absoluto: {abs(valor_esperado - valor_obtenido):.4f}")


def main():
    """Ejecuta todos los casos de prueba."""
    print("Ejecutando casos de prueba para el diagnóstico cardíaco...")
    test_persona_mayor_presion_alta()
    test_persona_joven_sedentaria()
    test_persona_adulta_sintomas()


if __name__ == '__main__':
    main()