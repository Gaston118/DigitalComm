import numpy as np

###########################################################################
#                                                                         #
#  ██████  ███████  ██████ ███████ ██████   ██████ ██  ██████  ███    ██  #
#  ██   ██ ██      ██      ██      ██   ██ ██      ██ ██    ██ ████   ██  #
#  ██████  █████   ██      █████   ██████  ██      ██ ██    ██ ██ ██  ██  #
#  ██   ██ ██      ██      ██      ██      ██      ██ ██    ██ ██  ██ ██  #
#  ██   ██ ███████  ██████ ███████ ██       ██████ ██  ██████  ██   ████  #
#                                                                         #
###########################################################################

#===========================================================================================
#                               DEMODULACIÓN DE SÍMBOLOS
#===========================================================================================
def nTuple_former(received_block, M, B, T):
    k = np.arange(M)
    down_chirp = np.exp(-1j * 2 * np.pi * (k * T * B) * k / M)
    ref = received_block * down_chirp
    spec = np.fft.fft(ref)
    return int(np.argmax(np.abs(spec)))

def decode_symbols_to_bits(symbols, SF):
    n_sym = len(symbols)
    return np.array([[(symbols[i] >> (SF-1-j)) & 1 for j in range(SF)] for i in range(n_sym)]).ravel()

#===========================================================================================
#                               REFERENCIAS DE CHIRP
#===========================================================================================
def make_down_ref(M, B, T):
    k = np.arange(M)
    return np.exp(-1j * 2 * np.pi * (k * T * B) * k / M) / np.sqrt(M)

#===========================================================================================
#                               FUNCIÓN DECHIRP (con ZP + peak merging)
#===========================================================================================
#           Dechirp + FFT con zero-padding + peak-merging.
#           Retorna:
#               ft_abs: magnitud plegada
#               pk_zp: índice entero del pico en resolución ZP (0..M*ZP-1)
#               pk_base_round: índice base M por redondeo (0..M-1)
#===========================================================================================
def dechirp(signal, start_idx, Ns, M, zp_factor, ref):
    seg = signal[start_idx : start_idx + Ns]
    if len(seg) < Ns:
        return None, None, None

    dechirped = seg * np.conj(ref)
    Nfft = int(Ns * zp_factor)
    spec = np.fft.fft(dechirped, n=Nfft)

    bin_num = int(M * zp_factor)
    pos = np.abs(spec[:bin_num])
    neg = np.abs(spec[-bin_num:])
    ft_abs = pos + neg              # PEAK MERGING

    pk_zp = int(np.argmax(ft_abs))
    pk_base_round = int(np.round(pk_zp / zp_factor)) % M
    return ft_abs, pk_zp, pk_base_round

#===========================================================================================
#                               DETECCIÓN DE PREÁMBULO
#===========================================================================================
def detect(signal, start_idx, Ns, preamble_len, M, zp, up_ref, mag_threshold=None):
    ii = int(start_idx)
    pk_bases = np.array([], dtype=int)
    sig_len = len(signal)
    bin_num = int(M * zp)

    while ii < sig_len - Ns * preamble_len:
        # Se detectaron suficientes upchirps
        if pk_bases.size == preamble_len - 1:
            x = ii - round((pk_bases[-1])/zp)
            return max(0, x)

        # Dechirp del símbolo actual
        mag, pk_zp, pk_base = dechirp(signal, ii, Ns, M, zp, up_ref)
        if mag is None:
            return -1
        # Umbral opcional
        if mag_threshold is not None and mag[pk_zp] < mag_threshold:
            pk_bases = np.array([], dtype=int)
            ii += Ns
            continue

        # Verificar coherencia entre bins consecutivos
        if pk_bases.size > 0:
            diff = (pk_bases[-1] - pk_base) % bin_num
            if diff > bin_num / 2:
                diff = bin_num - diff
            if diff <= max(1, int(zp)):
                pk_bases = np.append(pk_bases, pk_base)
            else:
                pk_bases = np.array([pk_base], dtype=int)
        else:
            pk_bases = np.array([pk_base], dtype=int)

        ii += Ns

    return -1

#===========================================================================================
#                               SINCRONIZACIÓN (WINDOW ALIGNMENT)
#===========================================================================================
#   Sincronización de paquete LoRa con zero-padding (ZP):
#       - Window alignment con down-chirp y pico firmado en resolución ZP.
#       - preamble_bin entero (base M) por redondeo de pk_u_zp/ZP.
#       - preamble_bin_zp (extendido) = pk_u_zp para demod ZP por símbolo.
#       - CFO continuo (Hz) estimado a partir del índice extendido pk_u_zp.
#       - Inicio de datos x_sync con 1.25 o 2.25 símbolos según SFD.
#       - Promedio de pk_zp en varios up-chirps del preámbulo (preamb_ns_to_avg).
#       - Refinamiento parabólico (sub-bin) alrededor del pico (log-parabola).

#       - Parámetros adicionales:
#           - preamble_ns_to_avg: tupla/lista con offsets (en símbolos) a usar para promediar
#                            p. ej. (4,5,6,7) usa 4..7 symbols antes de x_aligned.
#           - use_weighted_avg: si True pondera el promedio por la magnitud del pico.
#       
#       Retorna:
#           x_sync, preamble_bin, preamble_bin_zp, cfo_hz
#===========================================================================================
def sync(signal, x_detect, Ns, M, zp, up_ref, down_ref, bw, 
         preamble_ns_to_avg=(4,5,6,7)):
    """
    Window alignment siguiendo LoRaPHY.m
    
    Args:
        rf_freq: Frecuencia portadora (Hz) para calcular SFO
        bw: Ancho de banda (Hz)
    """
    L = len(signal)
    if x_detect is None or x_detect < 0:
        return -1, None, None, None

    bin_num = int(M * zp)

    # 1) Buscar primer downchirp
    x = int(x_detect)
    found = False
    while x < L - Ns:
        mag_u, pk_u_zp, _ = dechirp(signal, x, Ns, M, zp, up_ref)
        mag_d, pk_d_zp, _ = dechirp(signal, x, Ns, M, zp, down_ref)
        if mag_u is None or mag_d is None:
            break
        if abs(mag_d[int(pk_d_zp)]) > abs(mag_u[int(pk_u_zp)]):
            found = True
        x += Ns
        if found:
            break

    if not found:
        return -1, None, None, None

    # 2) Up-Down Alignment
    mag_d, pk_d_zp, _ = dechirp(signal, x, Ns, M, zp, down_ref)
    if mag_d is None:
        return -1, None, None, None

    # Convertir bin a offset en muestras (con signo)
    if pk_d_zp > bin_num // 2:
        to_samples = int(np.round((pk_d_zp - bin_num) / float(zp)))
    else:
        to_samples = int(np.round(pk_d_zp / float(zp)))
    
    x = x + to_samples
    x = max(0, min(L - Ns, x))

    # 3) Promedio de preamble bins (método circular)
    idxs = [x - a * Ns for a in preamble_ns_to_avg if 0 <= x - a * Ns <= L - Ns]
    if len(idxs) == 0:
        return -1, None, None, None

    pk_zps = []
    weights = []
    for idx in idxs:
        mag_u, pk_u_zp, _ = dechirp(signal, idx, Ns, M, zp, up_ref)
        if mag_u is None:
            continue
        pk_zps.append(float(pk_u_zp))
        weights.append(float(np.abs(mag_u[int(pk_u_zp)])))

    if len(pk_zps) == 0:
        return -1, None, None, None

    # Promedio circular con pesos normalizados
    pk_zps = np.array(pk_zps)
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalizar
    
    angles = 2.0 * np.pi * (pk_zps / bin_num)
    vec = np.sum(weights[:, np.newaxis] * np.exp(1j * angles[:, np.newaxis]), axis=0)
    mean_angle = np.angle(vec)
    mean_pk_zp = (mean_angle / (2.0 * np.pi)) * bin_num
    if mean_pk_zp < 0:
        mean_pk_zp += bin_num

    # Refinamiento parabólico en el mejor índice
    best_idx = idxs[np.argmax(weights * len(weights))]  # Desnormalizar para elegir
    mag_best, _, _ = dechirp(signal, best_idx, Ns, M, zp, up_ref)
    if mag_best is not None:
        pk_int = int(np.round(mean_pk_zp)) % bin_num
        delta = parabolic_refine(np.abs(mag_best[:bin_num]), pk_int)
        mean_pk_zp = (mean_pk_zp + delta) % bin_num

    preamble_bin_zp = mean_pk_zp
    preamble_bin = int(np.round(mean_pk_zp / zp)) % M

    # CFO calculation
    signed_pk = preamble_bin_zp if preamble_bin_zp <= bin_num/2 else preamble_bin_zp - bin_num
    cfo_hz = (signed_pk / zp / M) * bw

    # 4) Determinar inicio de datos (SFD alignment)
    x_prev = x - Ns
    if x_prev < 0:
        return -1, None, None, None

    mag_u, pk_u_zp, _ = dechirp(signal, x_prev, Ns, M, zp, up_ref)
    mag_d, pk_d_zp, _ = dechirp(signal, x_prev, Ns, M, zp, down_ref)
    if mag_u is None or mag_d is None:
        return -1, None, None, None

    if abs(mag_u[int(pk_u_zp)]) > abs(mag_d[int(pk_d_zp)]):
        x_sync = x + int(np.round(2.25 * Ns))
    else:
        x_sync = x + int(np.round(1.25 * Ns))

    return x_sync, preamble_bin, preamble_bin_zp, cfo_hz


def demod_data(tx_signal, data_start, num_data_symbols, M, ZP, up_ref, 
               preamble_bin_zp, cfo_hz, rf_freq):
    """
    Demodulación con compensación SFO y refinamiento parabólico
    """
    total_avail = len(tx_signal) - data_start
    Nwin = min(total_avail // M * M, num_data_symbols * M)
    if Nwin == 0:
        return np.array([], dtype=int), 0

    data_signal = tx_signal[data_start : data_start + Nwin]
    num_avail = Nwin // M
    bin_num = M * ZP

    # Calcular drift por SFO (como en LoRaPHY.m)
    sfo_drift = np.arange(1, num_avail + 1) * M * cfo_hz / rf_freq

    symbols_rx = []
    for i in range(num_avail):
        start_i = i * M
        ft_abs, pk_zp, _ = dechirp(data_signal, start_i, M, M, ZP, up_ref)
        if ft_abs is None:
            break

        # Refinamiento parabólico
        pk_refined = float(pk_zp) + parabolic_refine(ft_abs[:bin_num], int(pk_zp))

        # Compensar CFO y SFO
        sym_float = (pk_refined - preamble_bin_zp) / float(ZP)
        sym_compensated = sym_float - sfo_drift[i]
        
        sym = int(np.round(sym_compensated)) % M
        symbols_rx.append(sym)

    return np.array(symbols_rx, dtype=int), len(symbols_rx)

def parabolic_refine(mag, k):
        N = len(mag)
        if N < 3:
            return 0.0
        km = (k - 1) % N
        kp = (k + 1) % N

        y_m = np.log(mag[km] + 1e-12)
        y_0 = np.log(mag[k] + 1e-12)
        y_p = np.log(mag[kp] + 1e-12)
        denom = (y_m - 2.0 * y_0 + y_p)
        if abs(denom) < 1e-12:
            return 0.0
        delta = 0.5 * (y_m - y_p) / denom

        if delta > 0.5:
            delta = 0.5
        elif delta < -0.5:
            delta = -0.5
        return float(delta)