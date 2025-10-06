import numpy as np
from LoRaPHY import hamming_encode, interleaver, whitening, gray_encode

def bytes_to_bits(byte_array):
    """Convert bytes to bits (LSB first)."""
    bits = []
    for byte in byte_array:
        for i in range(8):
            bits.append((byte >> i) & 1)
    return np.array(bits, dtype=np.uint8)


def lora_encode(payload, SF=7, CR=4/7, LDRO=False, IH=1, CRC=0, verbose=False):
    """
    Complete LoRa encoding pipeline.
    
    Parameters
    ----------
    payload : array-like
        Input data bytes
    SF : int
        Spreading Factor (7-12)
    CR : float
        Coding Rate (4/5, 4/6, 4/7, 4/8)
    LDRO : bool
        Low Data Rate Optimization
    verbose : bool
        Print intermediate steps
        
    Returns
    -------
    np.ndarray
        Encoded LoRa symbols ready for modulation
    """
    payload = np.array(payload, dtype=np.uint8)
    
    # Step 1: Convert to bits
    payload_bits = bytes_to_bits(payload)
    if verbose:
        print(f"Bits de entrada: {len(payload_bits)} bits")
    
    # Step 2: Whitening
    whitened_payload = whitening(payload)
    whitened_bits = bytes_to_bits(whitened_payload)
    if verbose:
        print(f"Después de whitening: {len(whitened_bits)} bits")
    
    # Step 3: Hamming Encoding
    encoded_bits = hamming_encode(whitened_bits, CR)
    if verbose:
        print(f"Después de Hamming: {len(encoded_bits)} bits")
    
    # Step 4: Padding
    cr_map = {4/5: 5, 4/6: 6, 4/7: 7, 4/8: 8}
    symbols_per_block = cr_map[CR]
    #symbols_per_block = SF
    bits_per_symbol = SF - 2 if LDRO else SF
    bits_per_block = symbols_per_block * bits_per_symbol
    
    remainder = len(encoded_bits) % bits_per_block
    if remainder != 0:
        padding_needed = bits_per_block - remainder
        encoded_bits = np.concatenate([encoded_bits, np.zeros(padding_needed, dtype=np.uint8)])
        if verbose:
            print(f"Agregados {padding_needed} bits de padding")
    
    if verbose:
        print(f"Después de padding: {len(encoded_bits)} bits")
    
    # Step 5: Interleaving
    symbols = interleaver(encoded_bits, SF, CR, LDRO)
    if verbose:
        print(f"Después de interleaving: {len(symbols)} símbolos")
    
    # Step 6: Gray Encoding
    final_symbols = gray_encode(symbols, SF)
    
    if verbose:
        print(f"Símbolos finales listos para modular: {len(final_symbols)} símbolos")
    
    return np.array(final_symbols, dtype=np.uint16)
