import random

def lanzar_dado():
    return random.randint(1, 6)

num_lanzamientos = 1000000

def calcular(num_lanzamientos):
    conteo = 0
    for _ in range(num_lanzamientos):
        dado1 = lanzar_dado()
        dado2 = lanzar_dado()
        if (dado1+dado2) < 5:
            conteo += 1

    probabilidad = conteo / num_lanzamientos
    return probabilidad


probabilidad = calcular(num_lanzamientos)
print("La probabilidad es:", probabilidad)
