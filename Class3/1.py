# GENERAR UN SCRIPT QUE SIMULE UNA GENERACION DE UNA VARIABLE ALEATORIA GAUSSIANA.

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Definimos los parámetros de la distribución normal
mu = 0      # media
sigma = 0.4   # desviación estándar
# Generamos una muestra aleatoria de tamaño 1000
n = 10000000
muestra = np.random.normal(mu, sigma, n)
# Graficamos el histograma de la muestra
plt.hist((muestra), bins=100, density=True, alpha=0.5, color='g')
# Graficamos la función de densidad de probabilidad
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 1000)
p = stats.norm.pdf((x), mu, sigma)
plt.plot(x, p, 'k', linewidth=1)
max_pdf = stats.norm.pdf(mu, mu, sigma)

# Agregar una línea horizontal en el máximo de la PDF
plt.axhline(max_pdf, color='r', linestyle='--', linewidth=2, label='Máximo PDF')


plt.title('Histograma de la muestra y PDF de la distribución normal')
plt.xlabel('Valor')
plt.ylabel('Densidad de probabilidad')
plt.show()
# Este script genera una variable aleatoria gaussiana y la grafica junto con su PDF.
# La variable aleatoria sigue una distribución normal con media 0 y desviación estándar 1.
# Se generan 1000 muestras aleatorias y se grafican en un histograma.
# La función de densidad de probabilidad (PDF) se superpone al histograma para mostrar la forma de la distribución normal.
# La gráfica resultante muestra la distribución de la variable aleatoria generada.
# Se utiliza la biblioteca numpy para generar números aleatorios y matplotlib para graficar.
# Se utiliza scipy.stats para calcular la función de densidad de probabilidad.
