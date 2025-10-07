import numpy as np
from LoRaPHY import lora_encode

payload = np.array([0x48, 0x65, 0x6C, 0x6C, 0x6F], dtype=np.uint8)  # "Hello"
DE = 1
IH = 0
CRC = 1

SF_values = [7, 8, 9, 10, 11, 12]
CR_values = [4/5, 4/6, 4/7, 4/8]

print("\n=== TEST COMPLETO DE LORA ENCODE (SF x CR) ===\n")

for SF in SF_values:
    for CR in CR_values:
        print(f"\n🧩 Probando SF={SF}, CR={CR:.3f}, DE={DE}, IH={IH}, CRC={CRC}")
        try:
            encoded = lora_encode(payload, SF, CR, DE, IH, CRC, verbose=False)
            print(f"✅ {len(encoded)} símbolos generados correctamente.\n")
        except Exception as e:
            print(f"❌ ERROR para SF={SF}, CR={CR:.3f}: {e}\n")
