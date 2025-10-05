import numpy as np

def interleaver(bit_seq, SF, CR=4/7, LDRO=False):
    """
    LoRa diagonal interleaver.
    
    Parameters
    ----------
    bit_seq : array-like
        Bits after Hamming encoding
    SF : int
        Spreading Factor (7-12)
    CR : float
        Coding rate (4/5, 4/6, 4/7, 4/8)
    LDRO : bool
        Low Data Rate Optimization enabled
        
    Returns
    -------
    np.ndarray
        Array of symbol values (0 to 2^SF - 1)
    """
    bit_seq = np.array(bit_seq, dtype=np.uint8)
    
    # Bits per symbol
    bits_per_symbol = SF - 2 if LDRO else SF
    
    # Symbols per block
    cr_map = {4/5: 5, 4/6: 6, 4/7: 7, 4/8: 8}
    symbols_per_block = cr_map[CR]
    
    # Total bits needed per block
    bits_per_block = symbols_per_block * bits_per_symbol
    
    # Check length
    if len(bit_seq) % bits_per_block != 0:
        raise ValueError(f"Bit sequence length must be multiple of {bits_per_block}")
    
    n_blocks = len(bit_seq) // bits_per_block
    symbols = []
    
    for block_idx in range(n_blocks):
        # Extract block bits
        block_bits = bit_seq[block_idx * bits_per_block : 
                             (block_idx + 1) * bits_per_block]
        
        # Reshape to matrix: each row is the bits for one codeword position
        # Shape: (bits_per_symbol, symbols_per_block)
        matrix = block_bits.reshape(bits_per_symbol, symbols_per_block)
        
        # Apply diagonal interleaving
        # For each output symbol j
        for j in range(symbols_per_block):
            symbol_bits = np.zeros(bits_per_symbol, dtype=np.uint8)
            
            # For each bit position i in the symbol
            for i in range(bits_per_symbol):
                # Diagonal mapping: c_{j,i} = b_{i, (i+j) % bits_per_symbol}
                source_bit_pos = i
                source_symbol_pos = (i + j) % symbols_per_block
                symbol_bits[i] = matrix[source_bit_pos, source_symbol_pos]
            
            # Convert bits to symbol value (LSB first)
            symbol_value = 0
            for i, bit in enumerate(symbol_bits):
                symbol_value += bit * (2 ** i)
            
            symbols.append(symbol_value)
    
    return np.array(symbols, dtype=np.uint16)