import numpy as np
import matplotlib.pyplot as plt

def lanzar_tres_monedas():
    monedas = np.random.randint(0, 2, size=3)  # 0 representa "cara" y 1 representa "cruz"
    return np.sum(monedas)

num_muestras = 10000
muestras = [lanzar_tres_monedas() for _ in range(num_muestras)]


valores_unicos, conteos = np.unique(muestras, return_counts=True)
probabilidades = conteos / num_muestras
cdf = np.cumsum(probabilidades)

plt.step(valores_unicos, cdf, where='post')
plt.xlabel('Valor de la variable aleatoria')
plt.ylabel('Probabilidad acumulada')
plt.grid(True)
plt.show()
