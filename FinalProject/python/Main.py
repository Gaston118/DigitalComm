import numpy as np
import sys
import matplotlib.pyplot as plt

# ===========================================================================================
#                                       TRANSMISIÓN
# ===========================================================================================       

SF = 7                                              # Spreading Factor
M = 2**SF                                           # Número de bits por símbolo
B = 125e3                                           # Ancho de banda
T = 1/B                                             # Periodo de muestra
num_symbols = 200000                                # Número de símbolos a transmitir
num_bits = num_symbols * SF                         # Número de bits a transmitir
bits_tx = np.random.randint(0, 2, size=num_bits)    # Bits a transmitir

# ------------------------------------------------------------------------------------------
#                               CODIFICACIÓN DE BITS A SÍMBOLOS
# ------------------------------------------------------------------------------------------
def encode_bits_to_symbols(bits, SF):
    n_sym = len(bits) // SF
    return np.array([
        sum(bits[i*SF + j] << (SF-1-j) for j in range(SF))
        for i in range(n_sym)
    ])

#------------------------------------------------------------------------------------------
#                               FORMACIÓN DE LA FORMA DE ONDA
#------------------------------------------------------------------------------------------
def waveform_former(symbol, M, B, T):
    k = np.arange(M)
    phase = ((symbol + k) % M) * (k * T * B) / M
    chirp_waveform = np.exp(1j * 2 * np.pi * phase) / np.sqrt(M)
    return chirp_waveform

#------------------------------------------------------------------------------------------
#                           GENERACIÓN DE PREÁMBULO, NETID Y SFD
#------------------------------------------------------------------------------------------
def preamble_netid_sfd(M, B, T, preamble_len=8, netid_symbols=(24, 32)):
    up_chirp_0 = waveform_former(0, M, B, T)
    
    k = np.arange(M)
    down_chirp = np.exp(-1j * 2 * np.pi * (0 + k) * (k * T * B) / M)

    preamble = np.tile(up_chirp_0, preamble_len)
    
    netid = np.concatenate([waveform_former(s, M, B, T) for s in netid_symbols])
    
    chirp_len = len(up_chirp_0)
    sfd = np.concatenate([
        down_chirp,
        down_chirp,
        down_chirp[:chirp_len//4]
    ])
    
    return preamble, netid, sfd

#------------------------------------------------------------------------------------------
#                               GENERACION DE LA TRAMA LORA
#------------------------------------------------------------------------------------------
symbols_tx = encode_bits_to_symbols(bits_tx, SF)

def lora_modulate(symbols_tx, M, B, T):
    preamble, netid, sfd = preamble_netid_sfd(M, B, T)
    data_waveform = np.concatenate([waveform_former(i, M, B, T) for i in symbols_tx])
    return np.concatenate([preamble, netid, sfd, data_waveform])

tx_signal = lora_modulate(symbols_tx, M, B, T)

# ===========================================================================================
#                                       RECEPCIÓN
# ===========================================================================================       

#------------------------------------------------------------------------------------------
#                               FUNCIÓN DE DECHIRP
#------------------------------------------------------------------------------------------
def dechirp(signal, start_idx, Ns, M, zp_factor, upchirp):
    """
    Decodifica UN símbolo LoRa realizando dechirp + FFT.

    Parameters:
        signal     : señal recibida (time-domain)
        start_idx  : índice donde empieza el símbolo
        Ns         : cantidad de samples por símbolo
        M          : número de bins (=2^SF)
        zp_factor  : factor de zero-padding para FFT
        upchirp    : referencia de chirp base (ya precalculado)

    Returns:
        spectrum_abs : magnitud del espectro (opcional)
        peak_bin     : índice del bin con mayor energía
    """

    # 1. extraer símbolo
    segment = signal[start_idx : start_idx + Ns]
    if len(segment) < Ns:
        return None, None

    # 2. dechirp: multiplicar contra chirp conjugado
    dechirped = segment * np.conj(upchirp)

    # 3. FFT + zero padding
    Nfft = int(M * zp_factor)
    spectrum = np.fft.fft(dechirped, n=Nfft)

    # 4. quedarnos con la mitad (banda útil) y buscar pico
    spectrum_mag = np.abs(spectrum[:M])
    peak_bin = int(np.argmax(spectrum_mag))

    return spectrum_mag, peak_bin

#------------------------------------------------------------------------------------------
#                               DETECCION DE LA TRAMA LORA 
#------------------------------------------------------------------------------------------
def detect(signal, start_idx, Ns, preamble_len, M, zp, upchirp_ref, mag_threshold=None):
    ii = start_idx
    pk_bins = []
    sig_len = len(signal)
    
    while ii < sig_len - Ns * preamble_len:
        
        # condición de detección
        if len(pk_bins) == preamble_len - 1:
            offset = int(round(pk_bins[-1] / zp * 2))
            return max(0, ii - offset)

        # dechirp + fft
        mag, pk_bin = dechirp(signal, ii, Ns, M, zp, upchirp_ref)
        if mag is None:
            return -1
        
        # umbral opcional
        if mag_threshold is not None and mag[pk_bin] < mag_threshold:
            pk_bins = []
            ii += Ns
            continue
        
        # coherencia con el bin anterior
        if pk_bins:
            bin_diff = (pk_bins[-1] - pk_bin) % M
            if bin_diff > M/2:
                bin_diff = M - bin_diff

            if bin_diff <= zp:
                pk_bins.append(pk_bin)
            else:
                pk_bins = [pk_bin]
        else:
            pk_bins = [pk_bin]

        ii += Ns

    return -1

# Chirp de referencia para dechirp
up_ref = waveform_former(0, M, B, T)

zero_padding = 4  # Tolerancia en bins para detección coherente

x = detect(tx_signal, 0, M, 8, M, zero_padding, up_ref, mag_threshold=None)

if x != -1:
    preamble_start = x - (8 - 1) * M
    netid_len = 2 * M
    sfd_len = 2 * M + (M // 4)  
    data_start = preamble_start + 8 * M + netid_len + sfd_len

    # Asegurar que hay suficientes muestras para todos los símbolos
    num_data_symbols = num_symbols 
    expected_data_len = num_data_symbols * M
    if data_start + expected_data_len > len(tx_signal):
        # Si falta señal, ajusta num_data_symbols según lo disponible
        num_data_symbols = max(0, (len(tx_signal) - data_start) // M)
        print("Advertencia: señal corta. Ajustando número de símbolos a", num_data_symbols)

    print(f"Preambulo detectado = {x}")
    print(f"Inicio estimado del preámbulo = {preamble_start}")
    print(f"Inicio de datos = {data_start}, símbolos de datos a procesar = {num_data_symbols}")
else:
    print("No se detectó preámbulo")

# ------------------------------------------------------------------------------------------
#                               DEMODULACIÓN DE SÍMBOLOS
# ------------------------------------------------------------------------------------------

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

# --------------------------------------------------------------------------
#                  Extraer los datos desde la posición detectada
# --------------------------------------------------------------------------
data_signal = tx_signal[data_start : data_start + num_data_symbols*M]

symbols_rx = []
for i in range(num_data_symbols):
    block = data_signal[i*M : (i+1)*M]
    symbol_hat = nTuple_former(block, M, B, T)
    symbols_rx.append(symbol_hat)
symbols_rx = np.array(symbols_rx)

bits_rx = decode_symbols_to_bits(symbols_rx, SF)

# Calcular errores
num_symbol_errors = np.sum(symbols_tx[:num_data_symbols] != symbols_rx)
SER = num_symbol_errors / num_data_symbols

num_bit_errors = np.sum(bits_tx[:num_data_symbols*SF] != bits_rx)
BER = num_bit_errors / (num_data_symbols*SF)

print("\n================= RESULTADOS =================")
print(f"SF = {SF}, M = {M}")
print("Bits TX (primeros 20): ", bits_tx[:20])
print("Bits RX (primeros 20):", bits_rx[:20])
print("Símbolos TX (primeros 10):", symbols_tx[:10])
print("Símbolos RX (primeros 10):", symbols_rx[:10])
print(f"SER ideal: {SER}")
print(f"BER ideal: {BER}")

# ===========================================================================================
#                          SIMULACIÓN DE CANAL CON RUIDO
# ===========================================================================================

snr_dB_range = np.arange(-10, -1, 1)                    # Es/N0 (dB)
EsN0_dB_range = snr_dB_range + 10*np.log10(M)           # Para simular se suma, ya que SNR dB = Es/N0 - 10log10(M)
Es = 1                                                  # Energía por símbolo (normalizada)
BER_awgn = np.zeros_like(snr_dB_range, dtype=float)
SER_awgn = np.zeros_like(snr_dB_range, dtype=float)

h_freqsel    = np.array([np.sqrt(0.8), np.sqrt(0.2)])   # √0.8 δ[n] + √0.2 δ[n-1]
BER_freqsel  = np.zeros_like(snr_dB_range, dtype=float)
SER_freqsel  = np.zeros_like(snr_dB_range, dtype=float)

Eb = Es / SF                                            # Energía por bit

up_ref = waveform_former(0, M, B, T)

for idx, snr_dB in enumerate(EsN0_dB_range):

    print(f"\n===== Simulación SNR = {snr_dB_range[idx]} dB =====")

    SNR   = 10**(snr_dB / 10)               # relación lineal Potencia_señal / Potencia_ruido
    N0    = Es / SNR                        # densidad espectral de ruido
    sigma = np.sqrt(N0/2)                   # desviación típica por dimensión / estandar

    # -----------------------------------------------------------------------------------
    #                          CANAL AWGN PLANO
    # -----------------------------------------------------------------------------------

    # Generación de ruido AWGN complejo
    noise = sigma * (np.random.randn(len(tx_signal)) + 1j*np.random.randn(len(tx_signal)))

    # Señal recibida con AWGN
    rx_signal = tx_signal + noise

    x2 = detect(rx_signal, 0, M, 8, M, zero_padding, up_ref, mag_threshold=None)

    print("-------> Detección en canal AWGN plano <-------")

    if x2 != -1:
        preamble_start2 = x2 - (8 - 1) * M
        netid_len = 2 * M
        sfd_len = 2 * M + (M // 4)  
        data_start = preamble_start2 + 8 * M + netid_len + sfd_len

        # Asegurar que hay suficientes muestras para todos los símbolos
        num_data_symbols = num_symbols 
        expected_data_len = num_data_symbols * M
        if data_start + expected_data_len > len(tx_signal):
            # Si falta señal, ajusta num_data_symbols según lo disponible
            num_data_symbols = max(0, (len(tx_signal) - data_start) // M)
            print("Advertencia: señal corta. Ajustando número de símbolos a", num_data_symbols)

        print(f"Preambulo detectado = {x2} Inicio estimado del preámbulo = {preamble_start2}")
        print(f"Inicio de datos = {data_start}, símbolos de datos a procesar = {num_data_symbols}")
    else:
        print("❌ No se detectó preámbulo — saltando este punto SNR")
        continue

    data_signal_awgn = rx_signal[data_start : data_start + num_data_symbols*M]

    symbols_rx_awgn = []
    for i in range(num_data_symbols):
        block = data_signal_awgn[i*M : (i+1)*M]
        symbol_hat = nTuple_former(block, M, B, T)
        symbols_rx_awgn.append(symbol_hat)
    symbols_rx_awgn = np.array(symbols_rx_awgn)

    bits_rx_awgn = decode_symbols_to_bits(symbols_rx_awgn, SF)

    # Ajustar número de símbolos realmente procesados (seguro frente a señales cortas)
    num_proc_symbols = min(num_data_symbols, len(symbols_rx_awgn), len(symbols_tx))
    if num_proc_symbols == 0:
        print("Advertencia: no hay símbolos procesados para este punto SNR")
        SER_awgn[idx] = np.nan
        BER_awgn[idx] = np.nan
    else:
        # calcular SER y BER sólo sobre los símbolos/bits procesados
        SER_awgn[idx] = np.sum(symbols_tx[:num_proc_symbols] != symbols_rx_awgn[:num_proc_symbols]) / num_proc_symbols

        nbits_proc = num_proc_symbols * SF
        # asegurar que bits_rx_awgn tenga la longitud esperada (puede ocurrir que falten bits)
        bits_rx_awgn = bits_rx_awgn[:nbits_proc]
        BER_awgn[idx] = np.sum(bits_tx[:nbits_proc] != bits_rx_awgn) / nbits_proc

    print(f"SER (AWGN): {SER_awgn[idx]:.3e}, BER (AWGN): {BER_awgn[idx]:.3e}")

    # -----------------------------------------------------------------------------------
    #                          CANAL SELECTIVO EN FRECUENCIA
    # -----------------------------------------------------------------------------------

    # Se realiza ahora para la señal selectiva en frecuencia
    tx_faded = np.convolve(tx_signal, h_freqsel, mode='same')

    # AWGN con el mismo sigma ya calculado
    noise_sel     = sigma * (np.random.randn(len(tx_faded)) + 1j*np.random.randn(len(tx_faded)))
    rx_signal_sel = tx_faded + noise_sel

    x3 = detect(rx_signal_sel, 0, M, 8, M, zero_padding, up_ref, mag_threshold=None)

    print("-------> Detección en canal selectivo en frecuencia <-------")

    if x3 != -1:
        preamble_start3 = x3 - (8 - 1) * M
        netid_len = 2 * M
        sfd_len = 2 * M + (M // 4)  
        data_start3 = preamble_start3 + 8 * M + netid_len + sfd_len

        # Asegurar que hay suficientes muestras para todos los símbolos
        num_data_symbols = num_symbols 
        expected_data_len = num_data_symbols * M
        if data_start3 + expected_data_len > len(tx_signal):
            # Si falta señal, ajusta num_data_symbols según lo disponible
            num_data_symbols = max(0, (len(tx_signal) - data_start3) // M)
            print("Advertencia: señal corta. Ajustando número de símbolos a", num_data_symbols)

        print(f"Preambulo detectado = {x3} Inicio estimado del preámbulo = {preamble_start3}")
        print(f"Inicio de datos = {data_start3}, símbolos de datos a procesar = {num_data_symbols}")
    else:
        print("❌ No se detectó preámbulo — saltando este punto SNR")
        continue

    data_signal_sel = rx_signal_sel[data_start3 : data_start3 + num_data_symbols*M]

    symbols_rx_sel = []
    for sym_idx in range(num_data_symbols):
        block_sel = data_signal_sel[sym_idx*M : (sym_idx+1)*M]
        symbols_rx_sel.append(nTuple_former(block_sel, M, B, T))
    symbols_rx_sel = np.array(symbols_rx_sel)

    # Decodificación a bits
    bits_rx_sel = decode_symbols_to_bits(symbols_rx_sel, SF)

    # Ajustar número de símbolos realmente procesados (seguro frente a señales cortas)
    num_proc_symbols = min(num_data_symbols, len(symbols_rx_sel), len(symbols_tx))
    if num_proc_symbols == 0:
        print("Advertencia: no hay símbolos procesados para este punto SNR")
        SER_freqsel[idx] = np.nan
        BER_freqsel[idx] = np.nan
    else:
        # calcular SER y BER sólo sobre los símbolos/bits procesados
        SER_freqsel[idx] = np.sum(symbols_tx[:num_proc_symbols] != symbols_rx_sel[:num_proc_symbols]) / num_proc_symbols

        nbits_proc = num_proc_symbols * SF
        # asegurar que bits_rx_sel tenga la longitud esperada (puede ocurrir que falten bits)
        bits_rx_sel = bits_rx_sel[:nbits_proc]
        BER_freqsel[idx] = np.sum(bits_tx[:nbits_proc] != bits_rx_sel) / nbits_proc

    print(f"SER (Freq-sel): {SER_freqsel[idx]:.3e}, BER (Freq-sel): {BER_freqsel[idx]:.3e}")
    
plt.figure()
plt.semilogy(snr_dB_range, BER_awgn,    'o-',  label='BER Flat FSCM')
#plt.semilogy(snr_dB_range, SER_awgn,    's-',  label='SER Flat FSCM')
plt.semilogy(snr_dB_range, BER_freqsel, 'o--', label='BER Freq-sel FSCM')
#plt.semilogy(snr_dB_range, SER_freqsel, 's--', label='SER Freq-sel FSCM')

plt.xlim(-12, 0)
plt.ylim(1e-5, 1e-1)
plt.xticks(np.arange(-12, 1, 1))
plt.grid(True, which='both', ls='--', lw=0.5)
plt.xlabel('SNR (dB)')
plt.ylabel('BER / SER')
plt.title('Uncoded BER / SER – Flat vs Freq-sel FSCM (SF = 7)')
plt.legend()
plt.show()