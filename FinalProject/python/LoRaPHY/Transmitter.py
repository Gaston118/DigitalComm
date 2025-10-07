import numpy as np
from LoRaPHY import (hamming_encode, interleaver, whitening_seq, gray_encode, gen_header, generate_crc)
import numpy as np

def calc_sym_num(plen, SF, CR=1, crc=False, has_header=True, ldr=False):
    """
    Calculate LoRa symbol number exactly like the MATLAB version.

    Parameters:
    -----------
    plen : int
        Payload length in bytes
    SF : int
        Spreading Factor (7..12)
    CR : int
        Coding rate parameter (1..4, corresponds to 4/5..4/8)
    crc : bool
        Use CRC (True/False)
    has_header : bool
        Explicit header (True) or not (False)
    ldr : bool
        Low data rate optimization

    Returns:
    --------
    sym_num : int
        Number of LoRa symbols
    """

    crc_val = 1 if crc else 0
    header_val = 1 if has_header else 0
    ldr_val = 1 if ldr else 0

    numerator = 2*plen - SF + 7 + 4*crc_val - 5*(1 - header_val)
    denominator = SF - 2*ldr_val

    nblocks = max(np.ceil(numerator / denominator), 0)

    sym_num = 8 + (4 + CR) * nblocks

    return int(sym_num)

def lora_encode(payload, SF=7, CR=4/7, LDRO=False, IH=True, CRC=False, verbose=False):
    payload = np.array(payload, dtype=np.uint8)
    plen1 = len(payload)

    # -------------------------------
    # Número de símbolos esperados
    # -------------------------------
    sym_map_CR = {4/5:1, 4/6:2, 4/7:3, 4/8:4}
    CR_int = sym_map_CR[CR]
    sym_num = calc_sym_num(plen1, SF, CR_int, CRC, not IH, LDRO)

    if verbose:
        print("=====================================================")
        print(f"Número de símbolos esperados: {sym_num}")
        print("=====================================================")

    # -------------------------------
    # Generar CRC
    # -------------------------------
    if CRC:
        crc_bytes = generate_crc(payload)
        payload = np.concatenate([payload, crc_bytes])
        if verbose:
            print(f"CRC calculado: {crc_bytes}")
            print(f"Payload con CRC: {payload}")

    plen = len(payload)
    # -------------------------------
    # Whitening (solo payload)
    # -------------------------------
    seq = whitening_seq(255)
    data_w = np.bitwise_xor(payload, seq[:plen])

    # -------------------------------
    # Pasar a Nibbles
    # -------------------------------
    CR_map_2 = {4/5: 1, 4/6: 2, 4/7: 3, 4/8: 4}
    CR_int_2 = CR_map_2[CR]
    nibble_num = int((SF - 2) + (sym_num -8)/(CR_int_2 + 4)*(SF - 2*LDRO))

    if not IH:  
        data_nibble_count = nibble_num - (SF - 2)
    else:  # Implicit header
        data_nibble_count = nibble_num

    pad_bytes = max(0, int(np.ceil((data_nibble_count - 2*len(data_w))/2)))
    data_w = np.concatenate([data_w, 0xFF*np.ones(pad_bytes, dtype=np.uint8)])

    data_n = np.zeros(int(data_nibble_count), dtype=np.uint8)
    for i in range(int(data_nibble_count)):
        idx = i // 2  # Python index empieza en 0
        if i % 2 == 0:
            data_n[i] = data_w[idx] & 0x0F  # LSB
        else:
            data_n[i] = (data_w[idx] >> 4) & 0x0F  # MSB

    if verbose:
        print("Payload en nibbles:", data_n)

    # -------------------------------
    # Generar Header (si es explícito)
    # -------------------------------
    if not IH:  # Si hay header explícito
        header_nibbles = gen_header(plen, CR_int, int(CRC))
        # Concatenar header al principio de data_n
        data_n = np.concatenate([header_nibbles, data_n])
        if verbose:
            print("Header generado (nibbles):", header_nibbles)
            print("Payload completo con header:", data_n)

    # -------------------------------
    # Hamming
    # -------------------------------
    data_h = hamming_encode(data_n, SF, CR)
    if verbose:
        print("Payload con Hamming:", data_h)

    codewords = np.array(data_h, dtype=np.uint8)
    # -------------------------------
    # Interleaving
    # -------------------------------
    ppm = SF - 2*int(LDRO)
    rdd = CR_int + 4 # rdd = 8

    # Bloque 1: Entrelazado de los primeros SF-2 nibbles (bits de PHY/Header)
    symbols_i = interleaver(codewords[:SF-2], rdd=8)

    # Bloque 2 en adelante: Entrelazado del payload restante
    for i in range(SF-2, len(codewords), ppm):
        block = codewords[i:i+ppm]
        block_interleaved = interleaver(block, rdd)
        symbols_i = np.concatenate([symbols_i, block_interleaved])

    if verbose:
        print("Símbolos después del interleaving:", symbols_i)

    # -------------------------------
    # Gray encoding
    # -------------------------------
    symbols_g = gray_encode(symbols_i, SF, LDRO)
    if verbose:
        print("Símbolos después del Gray encoding:", symbols_g)

    if len(symbols_g) != sym_num:
        print("\033[91mERROR: La cantidad de símbolos no coincide con la esperada.\033[0m")
        print(f"\033[91mCantidad esperada: {sym_num}, Cantidad obtenida: {len(symbols_g)}\033[0m")
    else:
        print(f"\033[92mCantidad de símbolos coincide con la esperada. {len(symbols_g)}\033[0m")


    return symbols_g

def waveform_former(symbol, M, B, T):
    k = np.arange(M)
    phase = ((symbol + k) % M) * (k * T * B) / M
    chirp_waveform = np.exp(1j * 2 * np.pi * phase) / np.sqrt(M)
    return chirp_waveform

def lora_generate_preamble_netid_sfd(M, B, T, preamble_len=8, netid_symbols=(24, 32)):
    # Up-chirp base (símbolo 0)
    up_chirp_0 = waveform_former(0, M, B, T)
    
    # Down-chirp base
    #down_chirp_0 = waveform_former(0, M, B, T)[::-1]  # invertir tiempo para down-chirp
    k = np.arange(M)
    down_chirp = np.exp(-1j * 2 * np.pi * (0 + k) * (k * T * B) / M)

    # Preambulo: preamble_len repeticiones de up-chirp base
    preamble = np.tile(up_chirp_0, preamble_len)
    
    # NetID: dos up-chirps con símbolos del ID de red
    netid = np.concatenate([waveform_former(s, M, B, T) for s in netid_symbols])
    
    # SFD: 2 down-chirps + 1/4 de down-chirp (0.25*M samples)
    chirp_len = len(up_chirp_0)
    sfd = np.concatenate([
        down_chirp,
        down_chirp,
        down_chirp[:chirp_len//4]
    ])
    
    return preamble, netid, sfd

def lora_modulate(symbols, M, B, T):
    """
    Modulate LoRa symbols into a complex baseband signal.

    Parameters:
    -----------
    symbols : array-like
        Array of LoRa symbols (integers from 0 to M-1)
    M : int
        Number of symbols (M = 2^SF)
    B : float
        Bandwidth in Hz
    T : float
        Symbol duration in seconds

    """
    preamble, netid, sfd = lora_generate_preamble_netid_sfd(M, B, T)
    data = np.concatenate([waveform_former(s, M, B, T) for s in symbols])
    tx_signal = np.concatenate([preamble, netid, sfd, data])

    return tx_signal

# ===============================================================================
# OTRA FORMA DE HACER EL CHIRP - PARECE LA CORRECTA 
# ===============================================================================
def waveform_former_lora(symbol, M, B, T, is_up=True, cfo=0.0, tdelta=0.0, tscale=1.0, fs=None):
    """
    Genera un chirp de LoRa para un símbolo, implementado similar al MATLAB LoRaPHY.chirp.

    Parámetros
    ----------
    symbol : int o float
        Índice del símbolo (h). Puede ser no entero; se redondeará internamente (comportamiento MATLAB).
    M : int
        Número de bins de frecuencia (N = 2^SF).
    B : float
        Ancho de banda (Hz).
    T : float
        Duración del símbolo (segundos). Normalmente T ~ N / B en LoRa.
    is_up : bool, opcional
        True para up-chirp, False para down-chirp. Por defecto True.
    cfo : float, opcional
        Offset de frecuencia portadora (Hz). Por defecto 0.
    tdelta : float, opcional
        Desplazamiento de tiempo (segundos). Por defecto 0.
    tscale : float, opcional
        Factor de escalado de tiempo. Por defecto 1.
    fs : float o None, opcional
        Frecuencia de muestreo (Hz). Si es None, fs = M / T (así samp_per_sym == M).

    Retorna
    -------
    y : np.ndarray (complejo)
        Muestras en banda base complejas para el símbolo chirp (vector 1-D).
    """

    # Número de bins de frecuencia
    N = int(M)
    if fs is None:
        # Elegir fs para que samp_per_sym == M cuando T sea la duración del símbolo
        fs = float(M) / float(T)

    # Número de muestras por símbolo (MATLAB: samp_per_sym = round(fs/bw*N))
    samp_per_sym = int(round(fs / B * N))

    # Manejar valor fraccionario del símbolo exactamente como MATLAB
    h_orig = float(symbol)
    h = int(round(h_orig))
    # Ajustar cfo debido al redondeo como hace MATLAB
    cfo = cfo + (h_orig - h) / N * B

    # Pendiente del chirp y frecuencia inicial
    if is_up:
        k = B / T
        f0 = -B / 2.0 + cfo
    else:
        k = -B / T
        f0 = B / 2.0 + cfo

    # Primer segmento: longitud proporcional a (N-h)
    K1 = int(round(samp_per_sym * (N - h) / N))
    t1 = (np.arange(0, K1 + 1) / fs) * tscale + tdelta  # longitud K1+1
    # Fase para el primer segmento
    if t1.size > 0:
        c1 = np.exp(1j * 2.0 * np.pi * (t1 * (f0 + k * T * h / N + 0.5 * k * t1)))
    else:
        c1 = np.array([], dtype=np.complex128)

    # Continuidad de fase: calcular phi como hace MATLAB
    if c1.size == 0:
        phi = 0.0
    else:
        phi = np.angle(c1[-1])

    # Segundo segmento: longitud proporcional a h
    K2 = int(round(samp_per_sym * h / N))
    if K2 > 0:
        t2 = (np.arange(0, K2) / fs) + tdelta
        c2 = np.exp(1j * (phi + 2.0 * np.pi * (t2 * (f0 + 0.5 * k * t2))))
    else:
        c2 = np.array([], dtype=np.complex128)

    # Concatenar segmentos: c1 (sin la última muestra) + c2
    if c1.size > 0:
        y = np.concatenate([c1[:-1], c2])  # MATLAB usaba 1:snum-1
    else:
        y = c2.copy()

    return y
# ===============================================================================