import numpy as np

def lfsr_whitening_sequence(length):
    """
    Generate LoRa whitening sequence using LFSR.
    Polynomial: x^8 + x^6 + x^5 + x^4 + 1
    
    Parameters
    ----------
    length : int
        Number of bytes needed
        
    Returns
    -------
    np.ndarray
        Whitening sequence (bytes)
    """
    # LFSR with polynomial x^8 + x^6 + x^5 + x^4 + 1
    # Taps at positions: 8, 6, 5, 4 (counting from 1)
    # In 0-indexed: positions 7, 5, 4, 3
    
    # Initial state (arbitrary non-zero, typically 0xFF)
    state = 0xFF
    sequence = []
    
    for _ in range(length):
        # Current state forms the whitening byte
        sequence.append(state)
        
        # Calculate feedback bit
        # XOR of bits at positions 7, 5, 4, 3 (0-indexed)
        bit7 = (state >> 7) & 1
        bit5 = (state >> 5) & 1
        bit4 = (state >> 4) & 1
        bit3 = (state >> 3) & 1
        
        feedback = bit7 ^ bit5 ^ bit4 ^ bit3
        
        # Shift left and insert feedback at LSB
        state = ((state << 1) | feedback) & 0xFF
    
    return np.array(sequence, dtype=np.uint8)


def whitening(data_bytes, SF, CR=4/7):
    """
    LoRa whitening operation.
    
    According to paper: whitening happens BEFORE interleaving and Hamming encoding.
    The order is: Whitening → Hamming → Interleaving → Gray
    
    Parameters
    ----------
    data_bytes : array-like
        Input data bytes (header + payload + CRC if enabled)
    SF : int
        Spreading Factor (affects packet structure)
    CR : float
        Coding rate (4/5, 4/6, 4/7, 4/8)
        
    Returns
    -------
    np.ndarray
        Whitened bytes
    """
    data_bytes = np.array(data_bytes, dtype=np.uint8)
    
    # Generate whitening sequence
    # Note: header is NOT whitened according to paper (Section 4.6)
    # Only payload is whitened
    
    # The whitening sequence in LoRa is fixed (255 bytes max)
    whitening_seq = lfsr_whitening_sequence(255)
    
    # XOR data with whitening sequence
    n_bytes = len(data_bytes)
    whitened = data_bytes ^ whitening_seq[:n_bytes]
    
    return whitened
