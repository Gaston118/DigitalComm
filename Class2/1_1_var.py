import random
import matplotlib.pyplot as plt

def lanzar_moneda():
    return random.choice([0, 1])

num_lanzamientos = 10000000

def tres_monedas(num_lanzamientos, valor):
    conteo = 0
    for _ in range(num_lanzamientos):
        moneda1 = lanzar_moneda()
        moneda2 = lanzar_moneda()
        moneda3 = lanzar_moneda()
        if moneda1 + moneda2 + moneda3 == valor:
            conteo += 1

    probabilidad_evento = conteo / num_lanzamientos
    return probabilidad_evento

#probabilidad0 = tres_monedas(num_lanzamientos, 0)
#print("La probabilidad es:", probabilidad0)

#probabilidad0 = tres_monedas(num_lanzamientos, 1)
#print("La probabilidad es:", probabilidad0)

#probabilidad0 = tres_monedas(num_lanzamientos, 2)
#print("La probabilidad es:", probabilidad0)

#probabilidad0 = tres_monedas(num_lanzamientos, 3)
#print("La probabilidad es:", probabilidad0)

acumulado = 0
prob_acumulada = []
for valor in range(4):
    acumulado += tres_monedas(num_lanzamientos, valor)
    prob_acumulada.append(acumulado)

print("FDC:", prob_acumulada)
