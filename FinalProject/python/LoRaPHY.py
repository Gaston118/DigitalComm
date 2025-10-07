import numpy as np
from LoRaPHY import lora_encode, lora_modulate

payload = np.array([0x48, 0x65, 0x6C, 0x6C, 0x6F], dtype=np.uint8)  # "Hello"

SF = 10
CR = 4/8
DE = 1
IH = 0
CRC = 1

M = 2**SF
B = 125e3         # Ancho de banda
T = 1/B           # Periodo de muestra

symbols = lora_encode(payload, SF=SF, CR=CR, LDRO=DE, IH=IH, CRC=CRC, verbose=True)

symbols_modulated = lora_modulate(symbols, SF, B, T)
print("Señal modulada (primeros 10 valores):", symbols_modulated[:10])