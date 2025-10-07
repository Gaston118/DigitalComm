import numpy as np
from LoRaPHY import lora_encode

payload = np.array([0x48, 0x65, 0x6C, 0x6C, 0x6F], dtype=np.uint8)  # "Hello"

SF = 10
CR = 4/8
DE = 1
IH = 0
CRC = 0

symbols = lora_encode(payload, SF=SF, CR=CR, LDRO=DE, IH=IH, CRC=CRC, verbose=True)
print("Cantidad de símbolos generados:", len(symbols))