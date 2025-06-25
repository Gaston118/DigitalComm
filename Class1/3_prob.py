import random

def lanzar_moneda():
    return random.choice(['cara', 'cruz'])

num_lanzamientos = 10000000

def dos_monedas(num_lanzamientos):
    conteo = 0
    for _ in range(num_lanzamientos):
        moneda1 = lanzar_moneda()
        moneda2 = lanzar_moneda()
        if moneda1 == moneda2:
            conteo += 1
    probabilidad_evento = conteo / num_lanzamientos
    return probabilidad_evento

probabilidad= dos_monedas(num_lanzamientos)
print("La probabilidad es:", probabilidad)
