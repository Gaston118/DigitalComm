import random

def lanzar_moneda():
    return random.choice([0, 1])

def tres_monedas(num_lanzamientos):
    resultados = []
    for _ in range(num_lanzamientos):
        suma = lanzar_moneda() + lanzar_moneda() + lanzar_moneda()
        resultados.append(suma)
    
    valores_unicos = sorted(set(resultados))
    conteos = {valor: resultados.count(valor) for valor in valores_unicos}
    prob_acumulada = []
    acumulado = 0
    for valor in valores_unicos:
        acumulado += conteos[valor] / num_lanzamientos
        prob_acumulada.append(acumulado)
    
    def F_x(x):
        return sum(p for v, p in zip(valores_unicos, prob_acumulada) if x >= v)
    
    return valores_unicos, prob_acumulada, F_x

num_lanzamientos = 1000000
valores, F_x_vals, F_x = tres_monedas(num_lanzamientos)

print("Valores posibles:", valores)
print("FDC:", F_x_vals)
