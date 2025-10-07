import numpy as np

def generate_crc(data):
    """
    LoRa payload CRC (Polynomial: X^16 + X^12 + X^5 + 1)
    
    Retorna los 2 bytes del CRC calculado y XOReados con los últimos 2 bytes del payload.
    """
    data = np.array(data, dtype=np.uint8)
    
    if len(data) == 0:
        return np.array([0, 0], dtype=np.uint8)
    elif len(data) == 1:
        return np.array([data[-1], 0], dtype=np.uint8)
    elif len(data) == 2:
        return np.array([data[-1], data[-2]], dtype=np.uint8)
    
    input_data = data[:-2]
    
    poly = 0x1021
    crc = 0x0000
    
    for byte in input_data:
        crc ^= (int(byte) << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    
    # Separar en bytes
    crc_low = (crc >> 8) & 0xFF     # bits 15:8
    crc_high = crc & 0xFF           # bits 7:0
    
    # XOR con los últimos 2 bytes del payload
    checksum_b1 = crc_high ^ data[-1]
    checksum_b2 = crc_low ^ data[-2]
    
    return np.array([checksum_b1, checksum_b2], dtype=np.uint8)