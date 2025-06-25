import random

def lanzar_dado():
    return random.randint(1, 6)

num_lanzamientos = 10000000

def calcular_probabilidad_par(num_lanzamientos):
    conteo_pares = 0
    for _ in range(num_lanzamientos):
        resultado = lanzar_dado()
        if resultado % 2 == 0:
            conteo_pares += 1
    probabilidad_par = conteo_pares / num_lanzamientos
    return probabilidad_par


probabilidad = calcular_probabilidad_par(num_lanzamientos)
print("La probabilidad es:", probabilidad)
