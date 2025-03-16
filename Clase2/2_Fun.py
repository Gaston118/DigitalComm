import numpy as np

K = 0.0001  
alpha = 10000 

def fdp_exponencial(y):
    return K * np.exp(-y / alpha)

probabilidad = K * alpha * (1 - np.exp(-10000 / alpha))

print("La probabilidad es:", probabilidad)
