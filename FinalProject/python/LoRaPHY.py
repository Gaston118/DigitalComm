import numpy as np
from LoRaPHY import lora_encode

payload = np.array([0x48, 0x65, 0x6C, 0x6C, 0x6F], dtype=np.uint8)  # "Hello"

SF = 10
CR = 4/8
DE = 1
IH = 0
CRC = 1

symbols = lora_encode(payload, SF=SF, CR=CR, LDRO=DE, IH=IH, CRC=CRC, verbose=True)
print("Cantidad de símbolos generados:", len(symbols))

# Header generado (nibbles): [0 5 8 0 9] → bits: [0000 0101 1000 0000 1001]
# Primeros 2 nibbles: payload length = 5 → [0 5]
# Nibble 3: CR=4/8 (100) y Payload CRC enable = 0 → [8] (1000)
# Nibble 4: 3 bits reservados + 1er bit del checksum → [0]
# Nibble 5: bits 1-4 del CRC header → [9]