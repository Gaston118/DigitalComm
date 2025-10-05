import numpy as np

def gray_encode(symbols, SF):
    """
    Apply Gray encoding and +1 shift to symbol values.
    
    According to paper Section 4.4, LoRa applies:
    1. Gray encoding: gray_val = sym XOR (sym >> 1)
    2. Add 1: final_sym = (gray_val + 1) mod 2^SF
    
    Parameters
    ----------
    symbols : array-like
        Symbol values from interleaver (0 to 2^SF - 1)
    SF : int
        Spreading Factor (7-12)
        
    Returns
    -------
    np.ndarray
        Gray encoded symbols with +1 shift
    """
    symbols = np.array(symbols, dtype=np.uint16)
    final_symbols = []
    
    for sym in symbols:
        # Gray encoding
        gray_val = sym ^ (sym >> 1)
        
        # +1 shift (important: this is specific to LoRa)
        final_sym = (gray_val + 1) % (2**SF)
        
        final_symbols.append(final_sym)
    
    return np.array(final_symbols, dtype=np.uint16)