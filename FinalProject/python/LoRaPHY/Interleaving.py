import numpy as np

def interleaver(codewords, rdd):
    """
    Diagonal interleaving (LoRa PHY)
    
    Parameters
    ----------
    codewords : np.ndarray or list[int]
        Lista de codewords (nibbles). Longitud = P_b (SF-2 o ppm).
    rdd : int
        Número de bits por símbolo (por ejemplo, 5,6,7,8). Longitud = R_b (CR_int+4).
    
    Returns
    -------
    np.ndarray
        Símbolos después del interleaving. Longitud = rdd.
    """
    # 1. Convertir codewords a matriz de bits (right-msb)
    # Shape: (len(codewords), rdd) -> e.g., (5, 8)
    tmp = np.array([list(np.binary_repr(cw, width=rdd)) for cw in codewords], dtype=int)
    
    # 2. Circular shift diagonal
    # Cada columna (bit) se desplaza circularmente por su índice
    # La salida es una lista de arrays, cada uno de longitud len(codewords)
    shifted_cols = [np.roll(tmp[:,x], 1-(x+1)) for x in range(rdd)]
    
    # 3. Apilar columnas. Shape: (len(codewords), rdd) -> e.g., (5, 8)
    interleaved_matrix_cols = np.stack(shifted_cols, axis=1)
    
    # 4. <--- CORRECCIÓN CRÍTICA: Transponer la matriz.
    # Esto asegura que tengamos 'rdd' filas (símbolos) de longitud 'len(codewords)' (bits por símbolo).
    # Shape: (rdd, len(codewords)) -> e.g., (8, 5)
    interleaved_matrix = interleaved_matrix_cols.T
    
    # 5. Convertir fila de bits a int
    symbols_i = np.array([int("".join(map(str,row)),2) for row in interleaved_matrix], dtype=np.uint16)
    
    # La longitud de symbols_i ahora será 'rdd' (8)
    return symbols_i