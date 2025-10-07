import numpy as np

def bit_reduce_xor(nibble, positions):
    """Reduce bits in 'positions' (1-based) of nibble by XOR."""
    val = 0
    for pos in positions:
        val ^= (nibble >> (pos - 1)) & 1
    return val

def word_reduce_bitor(bits_and_shifts):
    """Combine values by OR."""
    val = 0
    for b in bits_and_shifts:
        val |= b
    return val

def hamming_encode(nibbles, sf=7, cr=4/8):
    """
    LoRa Hamming encoder (faithful to MATLAB version, CR as fraction 4/x).
    
    Parameters
    ----------
    nibbles : array-like of uint8
        Input data (4-bit values 0–15)
    sf : int
        Spreading factor
    cr : float
        Coding rate: 4/5, 4/6, 4/7, or 4/8
    
    Returns
    -------
    np.ndarray
        Encoded codewords (uint8)
    """

    cr_map = {4/5: 1, 4/6: 2, 4/7: 3, 4/8: 4}
    if cr not in cr_map:
        raise ValueError("CR must be one of 4/5, 4/6, 4/7, or 4/8")
    cr_int = cr_map[cr]
    
    nibbles = np.array(nibbles, dtype=np.uint8)
    codewords = np.zeros_like(nibbles)

    for i, nibble in enumerate(nibbles, start=1):
        # parity bits
        p1 = bit_reduce_xor(nibble, [1, 3, 4])
        p2 = bit_reduce_xor(nibble, [1, 2, 4])
        p3 = bit_reduce_xor(nibble, [1, 2, 3])
        p4 = bit_reduce_xor(nibble, [1, 2, 3, 4])
        p5 = bit_reduce_xor(nibble, [2, 3, 4])

        # CR for first SF-2 nibbles = 4/8
        cr_now = 4 if i <= sf - 2 else cr_int

        if cr_now == 1:  # 4/5
            codewords[i-1] = (p4 << 4) | nibble
        elif cr_now == 2:  # 4/6
            codewords[i-1] = word_reduce_bitor([
                p5 << 5,
                p3 << 4,
                nibble
            ])
        elif cr_now == 3:  # 4/7
            codewords[i-1] = word_reduce_bitor([
                p2 << 6,
                p5 << 5,
                p3 << 4,
                nibble
            ])
        elif cr_now == 4:  # 4/8
            codewords[i-1] = word_reduce_bitor([
                p1 << 7,
                p2 << 6,
                p5 << 5,
                p3 << 4,
                nibble
            ])
        else:
            raise ValueError("Invalid Code Rate")
    return codewords
