import numpy as np

def lfsr_whitening_sequence(length):
    """
    Genera secuencia de whitening para LoRa usando LFSR.
    Polinomio: x^8 + x^6 + x^5 + x^4 + 1
    
    Parámetros
    ----------
    length : int
        Número de bytes necesarios
        
    Retorna
    -------
    np.ndarray
        Secuencia de whitening (bytes)
    """

    state = 0xFF
    sequence = []
    
    for _ in range(length):
        sequence.append(state)
        
        # Calcular bit de feedback
        # XOR de bits en posiciones 7, 5, 4, 3 (índice 0)
        bit7 = (state >> 7) & 1
        bit5 = (state >> 5) & 1
        bit4 = (state >> 4) & 1
        bit3 = (state >> 3) & 1
        
        feedback = bit7 ^ bit5 ^ bit4 ^ bit3
        
        # Desplazar a la izquierda e insertar feedback en el LSB
        state = ((state << 1) | feedback) & 0xFF
    
    return np.array(sequence, dtype=np.uint8)


def whitening(data_bytes):
    """
    Operación de whitening para LoRa.
    
    Parámetros
    ----------
    data_bytes : array-like
        Bytes de datos de entrada (header + payload + CRC si está habilitado)
        
    Retorna
    -------
    np.ndarray
        Bytes con whitening aplicado
    """
    data_bytes = np.array(data_bytes, dtype=np.uint8)
    
    # Generar secuencia de whitening
    # Nota: el header NO tiene whitening según el paper (Sección 4.6)
    # Solo el payload tiene whitening
    
    # La secuencia de whitening en LoRa es fija (máximo 255 bytes)
    whitening_seq = lfsr_whitening_sequence(255)
    
    # Operación XOR entre datos y secuencia de whitening
    n_bytes = len(data_bytes)
    whitened = data_bytes ^ whitening_seq[:n_bytes]
    
    return whitened