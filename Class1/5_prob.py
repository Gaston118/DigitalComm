import random

def lanzar_dado():
    return random.randint(1, 6)

num_lanzamientos = 1000000

def calcular(num_lanzamientos):
    A = 0
    B = 0 
    for _ in range(num_lanzamientos):
        dado1 = lanzar_dado()
        dado2 = lanzar_dado()
        if (dado1+dado2) % 2 == 0:
            A += 1
    
            if (dado1+dado2) % 4 == 0:
                B += 1

    probabilidad_AB = B / num_lanzamientos
    probabilidad_B_A = B / A

    return probabilidad_AB, probabilidad_B_A
  

probabilidadAB, probabilidadA_B = calcular(num_lanzamientos)
print("P(AB):", probabilidadAB)
print("P(B/A):", probabilidadA_B)
