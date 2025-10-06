import numpy as np
from LoRaPHY import lora_encode

payload = np.array([0x48, 0x65, 0x6C, 0x6C, 0x6F], dtype=np.uint8)  # "Hello"

# Configuración
SF  = 7
CR  = 4/8
DE  = 0
IH  = 1     # Implicit Header
CRC = 0     # Sin CRC

# Codificación LoRa
encoded_symbols = lora_encode(payload, SF, CR, DE, IH, CRC, verbose=1)