import numpy as np

# Header generado (nibbles): [0 5 8 0 9] → bits: [0000 0101 1000 0000 1001]
# Primeros 2 nibbles: payload length = 5 → [0 5]
# Nibble 3: CR=4/8 (100) y Payload CRC enable = 0 → [8] (1000)
# Nibble 4: 3 bits reservados + 1er bit del checksum → [0]
# Nibble 5: bits 1-4 del CRC header → [9]

HEADER_CHECKSUM_MATRIX = np.array([
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1],
], dtype=np.uint8)


def gen_header(plen, cr, crc, header_checksum_matrix=HEADER_CHECKSUM_MATRIX):
    """
    Generate LoRa header (5 nibbles = 20 bits)
    MATLAB equivalent of gen_header()

    Parameters
    ----------
    plen : int
        Payload length (0–255)
    cr : int
        Coding rate index (1–4 → CR=4/5, 4/6, 4/7, 4/8)
    crc : int
        CRC enable flag (0 or 1)
    header_checksum_matrix : np.ndarray
        5x12 GF(2) checksum matrix

    Returns
    -------
    np.ndarray
        Header nibbles (array of length 5, dtype=uint8)
    """
    header_nibbles = np.zeros(5, dtype=np.uint8)

    # --- Primeros 3 nibbles ---
    header_nibbles[0] = (plen >> 4) & 0x0F  
    header_nibbles[1] = plen & 0x0F         
    header_nibbles[2] = (2 * cr) | (crc & 0x01)

    # Convertir cada nibble a 4 bits (MSB primero)
    bits_12 = np.zeros(12, dtype=np.uint8)
    for i in range(3):
        nibble_bits = np.array([
            (header_nibbles[i] >> 3) & 1,  # bit 3 (MSB)
            (header_nibbles[i] >> 2) & 1,  # bit 2
            (header_nibbles[i] >> 1) & 1,  # bit 1
            (header_nibbles[i] >> 0) & 1   # bit 0 (LSB)
        ], dtype=np.uint8)
        bits_12[i*4:(i+1)*4] = nibble_bits

    checksum_bits = np.mod(header_checksum_matrix @ bits_12, 2).astype(np.uint8)

    # --- Nibble 4: solo el primer bit del checksum ---
    header_nibbles[3] = checksum_bits[0]

    # --- Nibble 5: bits 1-4 del checksum ---
    header_nibbles[4] = 0
    for i in range(1, 5):  
        header_nibbles[4] |= checksum_bits[i] * (2 ** (4 - i))

    return header_nibbles
