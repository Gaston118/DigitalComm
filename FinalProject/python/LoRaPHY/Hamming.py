import numpy as np

def hamming_encode(bits, CR=4/8):
    """
    LoRa Hamming encoder.
    
    Parameters
    ----------
    bits : array-like
        Input bits (must be multiple of 4)
    CR : float
        Coding rate: 4/5, 4/6, 4/7, or 4/8
    
    Returns
    -------
    np.ndarray
        Encoded bits
    """
    bits = np.array(bits, dtype=np.uint8)
    assert len(bits) % 4 == 0, "Length must be multiple of 4"
    
    # Map CR to number of parity bits
    cr_map = {4/5: 1, 4/6: 2, 4/7: 3, 4/8: 4}
    assert CR in cr_map, "CR must be 4/5, 4/6, 4/7, or 4/8"
    n_parity = cr_map[CR]
    
    n_blocks = len(bits) // 4
    out = []
    
    for i in range(n_blocks):
        d1, d2, d3, d4 = bits[4*i : 4*i + 4]
        
        # Calculate parity bits (per paper equations 14-15)
        parities = []
        
        if n_parity >= 1:  # CR = 4/5: only p5
            p5 = d1 ^ d2 ^ d3
            parities.append(p5)
        
        if n_parity >= 2:  # CR = 4/6: p5, p1
            p1 = d1 ^ d2 ^ d4
            parities.append(p1)
        
        if n_parity >= 3:  # CR = 4/7: p5, p1, p2
            p2 = d1 ^ d3 ^ d4
            parities.append(p2)
        
        if n_parity == 4:  # CR = 4/8: p5, p1, p2, p3
            p3 = d2 ^ d3 ^ d4
            parities.append(p3)
        
        # LoRa format: parity bits (MSBs) then data bits (LSBs)
        codeword = parities + [d1, d2, d3, d4]
        out.extend(codeword)
    
    return np.array(out, dtype=np.uint8)