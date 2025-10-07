import numpy as np

def gray_encode(symbols_i, SF, LDRO=False):
    """
    Gray encoding with LoRa-specific adjustments.
    
    Parameters
    ----------
    symbols_i : array-like
        Symbols after interleaving (integers)
    SF : int
        Spreading Factor
    LDRO : bool
        Low Data Rate Optimization flag
        
    Returns
    -------
    np.ndarray
        Symbols after Gray encoding and LoRa-specific shifting
    """
    symbols_i = np.array(symbols_i, dtype=np.uint16)
    final_symbols = np.zeros_like(symbols_i, dtype=np.uint16)

    for i, sym in enumerate(symbols_i, start=1):  
        num = np.uint16(sym)
        mask = num >> 1
        while mask != 0:
            num ^= mask
            mask >>= 1

        if i <= 8 or LDRO:
            final_symbols[i-1] = (num * 4 + 1) % (2**SF)
        else:
            final_symbols[i-1] = (num + 1) % (2**SF)

    return final_symbols
