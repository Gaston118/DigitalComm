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
num_symbols = 250000                                # Número de símbolos a transmitir
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
def dechirp(signal, index, sample_num, bin_num, zero_padding, upchirp_ref):
    # Extraer el símbolo actual
    segment = signal[index:index + sample_num]
    if len(segment) < sample_num:
        return None, None

    # Dechirp
    dechirped = segment * np.conj(upchirp_ref)

    # FFT (con zero padding)
    fft_size = int(bin_num * zero_padding)
    spectrum = np.fft.fft(dechirped, n=fft_size)

    # Tomar la suma de las dos mitades
    half = int(min(bin_num, fft_size//2))
    ft_abs = np.abs(spectrum[:half]) + np.abs(spectrum[-half:])
    idx_peak = int(np.argmax(ft_abs))
    return ft_abs, idx_peak

#------------------------------------------------------------------------------------------
#                               DETECCION DE LA TRAMA LORA 
#------------------------------------------------------------------------------------------
def detect( signal, 
            start_index, 
            sample_num, 
            preamble_len, 
            bin_num, 
            zero_padding, 
            upchirp_ref, 
            mag_threshold=None):


    ii = int(start_index)
    pk_bin_list = [] # Almacena los índices de bin (posición del pico en la FFT) 
                     # Detectados en chirps sucesivos; usamos esta lista para verificar 
                     # que vemos chirps coherentes y consecutivos (el preámbulo está formado por chirps repetidos).
    sig_len = len(signal) # longitud total de la señal para evitar sobrescribir índices.

    while ii < sig_len - sample_num * preamble_len:
        # Se considera detectado el preámbulo cuando hemos observado preamble_len-1 chirps coherentes consecutivos.
        if len(pk_bin_list) == preamble_len - 1:
            # Alinear el pico del último chirp al inicio
            x = ii - int(round((pk_bin_list[-1]) / zero_padding * 2)) # redondea a la muestra entera más cercana.
            return max(0, x)  # asegurar índice no negativo
        
        # Ajusta la posición ii hacia atrás para alinear finamente el pico detectado con el principio del preámbulo.
        # pk_bin_list[-1] es el bin del pico del último chirp detectado.
        # Se divide por zero_padding y se multiplica por 2 porque la relación entre bin FFT y desplazamiento en muestras depende del muestreo y del padding.
        
        mag, pk_bin = dechirp(signal, ii, sample_num, bin_num, zero_padding, upchirp_ref)
        if mag is None:
            return -1

        # Filtrar por magnitud
        if mag_threshold is not None and mag[pk_bin] < mag_threshold:
            # Reiniciar y avanzar
            pk_bin_list = []
            ii += sample_num
            continue

            # Si ya había un bin previo (pk_bin_list[-1]), calculamos la distancia circular entre el bin anterior y el actual: usamos modulo bin_num porque el índice de bins es circular (frecuencia envuelta).
            # luego si la diferencia supera bin_num/2 usamos la distancia complementaria (la menor de las dos direcciones en el anillo).
            # Esto nos da la distancia mínima entre índices de pico en términos de bins.

        if pk_bin_list:
            bin_diff = (pk_bin_list[-1] - pk_bin) % bin_num
            if bin_diff > bin_num / 2:
                bin_diff = bin_num - bin_diff

            # Si el bin es coherente, agregarlo
            if bin_diff <= zero_padding:
                pk_bin_list.append(pk_bin)
            else:
                pk_bin_list = [pk_bin]
        else:
            pk_bin_list = [pk_bin]

        ii += sample_num

    return -1

# Chirp de referencia para dechirp
up_ref = waveform_former(0, M, B, T)

x = detect(tx_signal, 0, M, 8, M, 1, up_ref, mag_threshold=None)

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

snr_dB_range = np.arange(-11, 1, -2)                    # Es/N0 (dB)
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

    x2 = detect(rx_signal, 0, M, 8, M, 1, up_ref, mag_threshold=None)

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

    x3 = detect(rx_signal_sel, 0, M, 8, M, 1, up_ref, mag_threshold=None)

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