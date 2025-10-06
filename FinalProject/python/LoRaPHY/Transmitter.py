import numpy as np
from LoRaPHY import hamming_encode, interleaver, whitening_seq, gray_encode

import numpy as np

def calc_sym_num(plen, SF, CR=1, crc=False, has_header=True, ldr=False):
    """
    Calculate LoRa symbol number exactly like the MATLAB version.

    Parameters:
    -----------
    plen : int
        Payload length in bytes
    SF : int
        Spreading Factor (7..12)
    CR : int
        Coding rate parameter (1..4, corresponds to 4/5..4/8)
    crc : bool
        Use CRC (True/False)
    has_header : bool
        Explicit header (True) or not (False)
    ldr : bool
        Low data rate optimization

    Returns:
    --------
    sym_num : int
        Number of LoRa symbols
    """

    crc_val = 1 if crc else 0
    header_val = 1 if has_header else 0
    ldr_val = 1 if ldr else 0

    numerator = 2*plen - SF + 7 + 4*crc_val - 5*(1 - header_val)
    denominator = SF - 2*ldr_val

    nblocks = max(np.ceil(numerator / denominator), 0)

    sym_num = 8 + (4 + CR) * nblocks

    return int(sym_num)

def lora_encode(payload, SF=7, CR=4/7, LDRO=False, IH=True, CRC=False, verbose=False):
    payload = np.array(payload, dtype=np.uint8)
    plen = len(payload)

    # -------------------------------
    # Número de símbolos esperados
    # -------------------------------
    sym_map_CR = {4/5:1, 4/6:2, 4/7:3, 4/8:4}
    CR_int = sym_map_CR[CR]
    sym_num = calc_sym_num(plen, SF, CR_int, CRC, not IH, LDRO)

    if verbose:
        print("=====================================================")
        print(f"Número de símbolos esperados: {sym_num}")
        print("=====================================================")

    # -------------------------------
    # Whitening
    # -------------------------------
    seq = whitening_seq(255)
    data_w = np.bitwise_xor(payload, seq[:plen])
    if verbose:
        print("Payload blanqueado:", data_w)

    # -------------------------------
    # Pasar a Nibbles
    # -------------------------------
    CR_map_2 = {4/5: 1, 4/6: 2, 4/7: 3, 4/8: 4}
    CR_int_2 = CR_map_2[CR]
    nibble_num = int((SF - 2) + (sym_num -8)/(CR_int_2 + 4)*(SF - 2*LDRO))
    pad_bytes = int(np.ceil((nibble_num - 2*len(data_w))/2))
    data_w = np.concatenate([data_w, 0xFF*np.ones(pad_bytes, dtype=np.uint8)])

    data_n = np.zeros(int(nibble_num), dtype=np.uint8)
    for i in range(int(nibble_num)):
        idx = i // 2  # Python index empieza en 0
        if i % 2 == 0:
            data_n[i] = data_w[idx] & 0x0F  # LSB
        else:
            data_n[i] = (data_w[idx] >> 4) & 0x0F  # MSB

    if verbose:
        print("Payload en nibbles:", data_n)

    # -------------------------------
    # Hamming
    # -------------------------------
    data_h = hamming_encode(data_n, SF, CR)
    if verbose:
        print("Payload con Hamming:", data_h)

    codewords = np.array(data_h, dtype=np.uint8)
    # -------------------------------
    # Interleaving
    # -------------------------------
    ppm = SF - 2*int(LDRO)
    rdd = CR_int + 4 # rdd = 8

    # Bloque 1: Entrelazado de los primeros SF-2 nibbles (bits de PHY/Header)
    symbols_i = interleaver(codewords[:SF-2], rdd=8)

    # Bloque 2 en adelante: Entrelazado del payload restante
    for i in range(SF-2, len(codewords), ppm):
        block = codewords[i:i+ppm]
        block_interleaved = interleaver(block, rdd)
        symbols_i = np.concatenate([symbols_i, block_interleaved])

    if verbose:
        print("Símbolos después del interleaving:", symbols_i)

    # -------------------------------
    # Gray encoding
    # -------------------------------
    symbols_g = gray_encode(symbols_i, SF, LDRO)
    if verbose:
        print("Símbolos después del Gray encoding:", symbols_g)

    if len(symbols_g) != sym_num:
        print("WARNING: La cantidad de símbolos no coincide con la esperada.")
        print(f"Cantidad esperada: {sym_num}, Cantidad obtenida: {len(symbols_g)}")

    return symbols_g