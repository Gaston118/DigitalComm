import random

def lanzar_dado():
    return random.randint(1, 6)

num_lanzamientos = 1000000

#ENTIENDO QUE EL EJERCICIO PIDE QUE SE LANZEN DOS DADOS 
#Y SE CALCULE LA PROBABILIDAD DE QUE LA SUMA DE LOS DOS DADOS SEA PAR

def calcular_probabilidad_par(num_lanzamientos):
    conteo_pares = 0
    for _ in range(num_lanzamientos):
        dado1 = lanzar_dado()
        dado2 = lanzar_dado()
        if (dado1+dado2) % 2 == 0:
            conteo_pares += 1

    probabilidad_par = conteo_pares / num_lanzamientos
    return probabilidad_par


probabilidad = calcular_probabilidad_par(num_lanzamientos)
print("La probabilidad es:", probabilidad)
