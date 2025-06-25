import random

#ACA VOY A HACER QUE SE LANZEN N DADOS
def lanzar_dado(n):
    unos=0
    for _ in range(n):
        if random.randint(1, 6) == 1:
            unos += 1

    return unos==n

num_lanzamientos = 100000

def calcular(n, num_lanzamientos):
    conteo = 0
    for _ in range(num_lanzamientos):
        if lanzar_dado(n):
            conteo += 1

    probabilidad = conteo / num_lanzamientos
    return probabilidad

#ACA ELIJO 2 DADOS
probabilidad = calcular(2, num_lanzamientos)
print("La probabilidad es:", probabilidad)
