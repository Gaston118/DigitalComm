import numpy as np

def gray_encode(symbols, SF):
    """
    Aplica la codificación Gray y un desplazamiento +1 a los valores de los símbolos.
    
    Según la Sección 4.4 del paper, LoRa realiza:
    1. Codificación Gray: gray_val = sym XOR (sym >> 1)
    2. Sumar 1: final_sym = (gray_val + 1) mod 2^SF
    
    Parámetros
    ----------
    symbols : array-like
        Valores de los símbolos provenientes del interleaver (0 a 2^SF - 1)
    SF : int
        Factor de expansión (7-12)
        
    Retorna
    -------
    np.ndarray
        Símbolos codificados en Gray con el desplazamiento +1
    """
    symbols = np.array(symbols, dtype=np.uint16)
    final_symbols = []
    
    for sym in symbols:
        # Codificación Gray: XOR del símbolo con su versión desplazada un bit a la derecha
        gray_val = sym ^ (sym >> 1)
        
        # Desplazamiento +1 (específico de LoRa)
        final_sym = (gray_val + 1) % (2**SF)
        
        # Agregar el símbolo final a la lista
        final_symbols.append(final_sym)
    
    # Convertir la lista final a un array de numpy
    return np.array(final_symbols, dtype=np.uint16)
