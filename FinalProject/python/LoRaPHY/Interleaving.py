import numpy as np

def interleaver(bit_seq, SF, CR=4/7, LDRO=False):
    """
    Intercalador diagonal de LoRa.
    
    Parámetros
    ----------
    bit_seq : array-like
        Bits luego de la codificación Hamming
    SF : int
        Factor de expansión (7-12)
    CR : float
        Tasa de codificación (4/5, 4/6, 4/7, 4/8)
    LDRO : bool
        Optimización de baja tasa de datos activada
        
    Retorna
    -------
    np.ndarray
        Array de valores de símbolos (0 a 2^SF - 1)
    """
    bit_seq = np.array(bit_seq, dtype=np.uint8)
    
    # Cantidad de bits por símbolo
    bits_per_symbol = SF - 2 if LDRO else SF
    
    # Símbolos por bloque según la tasa de codificación
    cr_map = {4/5: 5, 4/6: 6, 4/7: 7, 4/8: 8}
    symbols_per_block = cr_map[CR]
    
    # Total de bits necesarios por bloque
    bits_per_block = symbols_per_block * bits_per_symbol
    
    # Verificar que la longitud de la secuencia sea múltiplo de bits por bloque
    if len(bit_seq) % bits_per_block != 0:
        raise ValueError(f"La longitud de la secuencia de bits debe ser múltiplo de {bits_per_block}")
    
    # Número de bloques a procesar
    n_blocks = len(bit_seq) // bits_per_block
    symbols = []
    
    for block_idx in range(n_blocks):
        # Extraer los bits del bloque actual
        block_bits = bit_seq[block_idx * bits_per_block : 
                             (block_idx + 1) * bits_per_block]
        
        # Redimensionar a matriz: cada fila contiene los bits de una posición del código
        # Forma: (bits_por_símbolo, símbolos_por_bloque)
        matrix = block_bits.reshape(bits_per_symbol, symbols_per_block)
        
        # Aplicar intercalado diagonal
        # Para cada símbolo de salida j
        for j in range(symbols_per_block):
            symbol_bits = np.zeros(bits_per_symbol, dtype=np.uint8)
            
            # Para cada posición de bit i en el símbolo
            for i in range(bits_per_symbol):
                # Mapeo diagonal: c_{j,i} = b_{i, (i+j) % bits_por_símbolo}
                source_bit_pos = i
                source_symbol_pos = (i + j) % symbols_per_block
                symbol_bits[i] = matrix[source_bit_pos, source_symbol_pos]
            
            # Convertir los bits a valor de símbolo (LSB primero)
            symbol_value = np.uint32(0)
            for i, bit in enumerate(symbol_bits):
                symbol_value += int(bit) * (2 ** i)
            
            # Agregar el símbolo final a la lista
            symbols.append(symbol_value)
    
    # Convertir la lista de símbolos a un array de numpy
    return np.array(symbols, dtype=np.uint32)
