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
    reference_chirp = received_block * down_chirp
    spectrum = np.fft.fft(reference_chirp)
    return np.argmax(np.abs(spectrum))

def decode_symbols_to_bits(symbols, SF):
    n_sym = len(symbols)
    return np.array([
        [(symbols[i] >> (SF-1-j)) & 1 for j in range(SF)]
        for i in range(n_sym)
    ]).flatten()

#===========================================================================================
#                           GENERAR DOWNCHIRP DE REFERENCIA
#===========================================================================================
def make_down_ref(M, B, T):
    k = np.arange(M)
    return np.exp(-1j * 2 * np.pi * (k * T * B) * k / M) / np.sqrt(M)

#===========================================================================================
#                               FUNCIÓN DE DECHIRP
#===========================================================================================
def dechirp(signal, start_idx, Ns, M, zp_factor, ref):
    seg = signal[start_idx : start_idx + Ns]
    if len(seg) < Ns:
        return None, None
    dechirped = seg * np.conj(ref)
    Nfft = int(Ns * zp_factor)  # usar Ns para FFT si zp>1
    spec = np.fft.fft(dechirped, n=Nfft)
    bin_num = int(M * zp_factor)
    # plegado (suma mitades) para robustez
    pos = np.abs(spec[:bin_num])
    neg = np.abs(spec[-bin_num:])
    ft_abs = pos + neg
    pk = int(np.argmax(ft_abs))
    # mapear el pico al bin base M
    pk_mod_M = pk % M
    return ft_abs, pk_mod_M

#===========================================================================================
#                               DETECCION DE LA TRAMA LORA
#===========================================================================================
def detect(signal, start_idx, Ns, preamble_len, M, zp, up_ref, mag_threshold=None):
    ii = int(start_idx)
    pk_bins = []
    sig_len = len(signal)
    while ii < sig_len - Ns * preamble_len:
        if len(pk_bins) == preamble_len - 1:
            offset_bin = pk_bins[-1]
            if offset_bin > M // 2:
                offset_bin -= M  # signed in [-M/2, M/2]
            offset = int(round(offset_bin * (Ns / M)))  # con Ns=M => offset=offset_bin
            return max(0, ii - offset)
        mag, pk_bin = dechirp(signal, ii, Ns, M, zp, up_ref)
        if mag is None:
            return -1
        if mag_threshold is not None and mag[pk_bin] < mag_threshold:
            pk_bins = []
            ii += Ns
            continue
        if pk_bins:
            diff = (pk_bins[-1] - pk_bin) % M
            if diff > M/2:
                diff = M - diff
            pk_bins = pk_bins + [pk_bin] if diff <= max(1, zp) else [pk_bin]
        else:
            pk_bins = [pk_bin]
        ii += Ns
    return -1

#===========================================================================================
#                           ESTIMACIÓN DE CFO A PARTIR DE LA FASE
#===========================================================================================
def estimate_cfo_phase(symbol_seg, up_ref, fs_eff):
    """
    Estima CFO continuo (Hz) a partir de la fase media tras dechirp:
    d[n] = symbol * conj(up_ref); r = sum(conj(d[n]) * d[n+1])
    phi = arg(r) ≈ 2π Δf / fs_eff  ->  Δf = phi * fs_eff / (2π)
    """
    d = symbol_seg * np.conj(up_ref)
    if len(d) < 2:
        return 0.0
    r = np.vdot(d[:-1], d[1:])             # correlación un paso
    phi = np.angle(r)                      # fase media por muestra
    return float(fs_eff) * float(phi) / (2.0 * np.pi)

#===========================================================================================
#                               FUNCIÓN DE SINCRONIZACIÓN
#===========================================================================================
def sync(signal, x_detect, Ns, M, zp, up_ref, down_ref, bw=None):
    L = len(signal)
    if x_detect is None or x_detect < 0:
        return -1, None, None

    # 1) buscar primer downchirp después de x_detect
    x = int(x_detect)
    found = False
    while x <= L - Ns:
        mag_u, pk_u = dechirp(signal, x, Ns, M, zp, up_ref)
        mag_d, pk_d = dechirp(signal, x, Ns, M, zp, down_ref)
        if mag_u is None or mag_d is None:
            break
        if mag_d[int(pk_d)] > mag_u[int(pk_u)]:
            found = True
            break
        x += Ns
    if not found:
        return -1, None, None

    # 2) avanzar un símbolo (como LoRaPHY)
    x = x + Ns
    if x > L - Ns:
        return -1, None, None

    # 3) up–down alignment en el downchirp
    mag_d, pk_d = dechirp(signal, x, Ns, M, zp, down_ref)
    if mag_d is None:
        return -1, None, None
    pkd = int(pk_d)
    pkd_wrap = pkd - M if pkd > M//2 else pkd
    to_samples = int(round(pkd_wrap * (Ns / M)))
    x_aligned = x + to_samples
    x_aligned = max(0, min(L - Ns, x_aligned))

    # 4) preamble_bin entero (para corrección por bin)
    xi = x_aligned - 4*Ns
    if xi < 0:
        xi = x_aligned - Ns
        if xi < 0:
            return -1, None, None
    mag_u_ref, pku = dechirp(signal, xi, Ns, M, zp, up_ref)
    if mag_u_ref is None:
        return -1, None, None
    preamble_bin = int(pku)  # 0..M-1

    # 5) CFO continuo (Hz) por fase, promediando varios up-chirps
    fs_eff = bw if bw is not None else 1.0  # en tu caso Ns=M -> fs_eff = B
    cfo_list = []
    for k in (1, 2, 3, 4):
        xi_k = x_aligned - (k+1)*Ns
        if xi_k < 0: break
        seg = signal[xi_k : xi_k + Ns]
        if len(seg) < Ns: break
        cfo_list.append(estimate_cfo_phase(seg, up_ref, fs_eff))
    if cfo_list:
        cfo_hz = float(np.median(cfo_list))
    else:
        # fallback al CFO por bin entero si no hay suficientes símbolos
        b_signed = preamble_bin - M if preamble_bin > M//2 else preamble_bin
        cfo_hz = (b_signed / M) * float(fs_eff)

    # 6) decidir 1.25 o 2.25 símbolos según SFD
    x_prev = x_aligned - Ns
    if x_prev < 0:
        return -1, None, None
    mag_up_prev, pk_up_prev = dechirp(signal, x_prev, Ns, M, zp, up_ref)
    mag_dn_prev, pk_dn_prev = dechirp(signal, x_prev, Ns, M, zp, down_ref)
    if mag_up_prev is None or mag_dn_prev is None:
        return -1, None, None

    if mag_up_prev[int(pk_up_prev)] > mag_dn_prev[int(pk_dn_prev)]:
        x_sync = x_aligned + int(round(2.25 * Ns))
    else:
        x_sync = x_aligned + int(round(1.25 * Ns))

    x_sync = max(0, min(L - 1, x_sync))
    return x_sync, preamble_bin, cfo_hz