import numpy as np

def whitening_seq(n_bytes):
    """
    Generate LoRa whitening sequence using 8-bit LFSR.
    Polynomial: x^8 + x^6 + x^5 + x^4 + 1

    Parameters
    ----------
    n_bytes : int
        Number of bytes to generate

    Returns
    -------
    np.ndarray
        Whitening sequence (uint8 array)
    """
    seq = np.zeros(n_bytes, dtype=np.uint8)
    lfsr = 0xFF  # initial value

    for i in range(n_bytes):
        seq[i] = lfsr
        # Compute feedback bit = XOR of bits 8,6,5,4 (0-based: 7,5,4,3)
        feedback = ((lfsr >> 7) ^ (lfsr >> 5) ^ (lfsr >> 4) ^ (lfsr >> 3)) & 1
        # Shift and insert feedback at LSB
        lfsr = ((lfsr << 1) & 0xFF) | feedback

    return seq
