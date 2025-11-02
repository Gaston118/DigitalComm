import numpy as np
import matplotlib.pyplot as plt
import Transimicion as tx
import Recepcion as rx

SF = 7                                              # Spreading Factor
M = 2**SF                                           # Número de bits por símbolo
B = 125e3                                           # Ancho de banda
T = 1/B                                             # Periodo de muestra
num_symbols = 2000                                  # Número de símbolos a transmitir
num_bits = num_symbols * SF                         # Número de bits a transmitir
bits_tx = np.random.randint(0, 2, size=num_bits)    # Bits a transmitir
zero_padding = 4                                    # Factor de zero padding en la FFT
num_data_symbols = num_bits // SF                   # Número de símbolos de datos

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

up_ref = tx.waveform_former(0, M, B, T)

N_FRAMES = 10                   # cantidad de tramas por cada SNR
idle_base = 3 * M               # longitud del "idle" o ruido previo
jitter_max_samples = 10 * M     # máximo jitter en detección

for idx, snr_dB in enumerate(EsN0_dB_range):

    BER_tmp = np.zeros(N_FRAMES)
    SER_tmp = np.zeros(N_FRAMES)
    BER_tmp_sel = np.zeros(N_FRAMES)
    SER_tmp_sel = np.zeros(N_FRAMES)

    print(f"\n===== Simulación SNR = {snr_dB_range[idx]} dB =====")

    for frame_idx in range(N_FRAMES):

        # -----------------------------------------------------------------------------------
        #                          CANAL AWGN PLANO
        # -----------------------------------------------------------------------------------

        # --- 1. Generar trama aleatoria
        bits_tx = np.random.randint(0, 2, size=num_bits)
        symbols_tx = tx.encode_bits_to_symbols(bits_tx, SF)
        tx_frame = tx.lora_modulate(symbols_tx, M, B, T)

        jitter = np.random.randint(0, jitter_max_samples + 1)  
        idle_len_frame = idle_base + jitter
        
        # --- 2. Generar ruido previo
        noise_idle = (np.random.randn(idle_len_frame) + 1j*np.random.randn(idle_len_frame)) / np.sqrt(2)
        noise_idle *= np.sqrt(np.mean(np.abs(tx_frame)**2))

         # --- 3. Concatenar idle + trama
        tx_signal = np.concatenate([noise_idle, tx_frame])

        # --- 4. Aplicar canal y ruido AWGN
        SNR   = 10**(snr_dB / 10)                   # relación lineal Potencia_señal / Potencia_ruido
        N0    = Es / SNR                            # densidad espectral de ruido
        sigma = np.sqrt(N0/2)                       # desviación típica por dimensión / estandar

        noise = sigma * (np.random.randn(len(tx_signal)) + 1j*np.random.randn(len(tx_signal)))

        # Señal recibida con AWGN
        rx_signal = tx_signal + noise

        # --- 5. Detección de preámbulo
        x_detect = rx.detect(rx_signal, 0, M, 8, M, zero_padding, up_ref, mag_threshold=None)

        if x_detect == -1:
            # si no detecta, contar como error total
            BER_tmp[frame_idx] = 1.0
            SER_tmp[frame_idx] = 1.0
            print("❌ No se detectó preámbulo AWGN")
            continue

        preamble_start = x_detect - (8 - 1) * M
        netid_len = 2 * M
        sfd_len = 2 * M + (M // 4)
        data_start = preamble_start + 8 * M + netid_len + sfd_len
        data_signal = rx_signal[data_start : data_start + num_data_symbols * M]

        print(f"Preambulo detectado = {x_detect} Inicio estimado del preámbulo = {preamble_start}")

        # --- 6. Demodulación símbolo a símbolo
        symbols_rx = []
        for i in range(num_data_symbols):
            block = data_signal[i*M : (i+1)*M]
            if len(block) < M:
                break
            symbols_rx.append(rx.nTuple_former(block, M, B, T))
        symbols_rx = np.array(symbols_rx)
        bits_rx = rx.decode_symbols_to_bits(symbols_rx, SF)

        # --- 7. Cálculo de errores
        #num_symbol_errors = np.sum(symbols_tx[:num_data_symbols] != symbols_rx[:num_data_symbols])
        #num_bit_errors = np.sum(bits_tx[:num_data_symbols*SF] != bits_rx[:num_data_symbols*SF])

        #SER_tmp[frame_idx] = num_symbol_errors / num_data_symbols
        #BER_tmp[frame_idx] = num_bit_errors / (num_data_symbols * SF)

        # número de símbolos realmente demodulados
        num_proc_symbols = len(symbols_rx)

        if num_proc_symbols == 0:
            SER_tmp[frame_idx] = 1.0
            BER_tmp[frame_idx] = 1.0
        else:
            # recortar el vector de símbolos transmitidos al número procesado
            tx_sym_chunk = symbols_tx[:num_proc_symbols]
            rx_sym_chunk = symbols_rx[:num_proc_symbols]

            # símbolo a símbolo
            num_symbol_errors = np.sum(tx_sym_chunk != rx_sym_chunk)
            SER_tmp[frame_idx] = num_symbol_errors / num_proc_symbols

            # ahora bits: recortar según símbolos realmente procesados
            nbits_proc = num_proc_symbols * SF
            tx_bits_chunk = bits_tx[:nbits_proc]
            rx_bits_chunk = rx.decode_symbols_to_bits(rx_sym_chunk, SF)  # si ya tienes bits_rx, recorta directamente
            # asegúrate rx_bits_chunk tiene al menos nbits_proc (it should)
            rx_bits_chunk = rx_bits_chunk[:nbits_proc]

            num_bit_errors = np.sum(tx_bits_chunk != rx_bits_chunk)
            BER_tmp[frame_idx] = num_bit_errors / nbits_proc

     # --- 8. Promedio de resultados sobre las N tramas
    SER_awgn[idx] = np.mean(SER_tmp)
    BER_awgn[idx] = np.mean(BER_tmp)

    print(f"SER promedio (AWGN): {SER_awgn[idx]:.3e}, BER promedio (AWGN): {BER_awgn[idx]:.3e}")

    for frame_idx in range(N_FRAMES):

        # -----------------------------------------------------------------------------------
        #                          CANAL SELECTIVO EN FRECUENCIA
        # -----------------------------------------------------------------------------------
        
        # --- 1. Generar trama aleatoria
        bits_tx = np.random.randint(0, 2, size=num_bits)
        symbols_tx = tx.encode_bits_to_symbols(bits_tx, SF)
        tx_frame = tx.lora_modulate(symbols_tx, M, B, T)

        jitter = np.random.randint(0, jitter_max_samples + 1)
        idle_len = idle_base + jitter

        # --- 2. Generar ruido previo (idle)
        noise_idle = (np.random.randn(idle_len) + 1j*np.random.randn(idle_len)) / np.sqrt(2)
        noise_idle *= np.sqrt(np.mean(np.abs(tx_frame)**2))  # igualar potencia promedio

        # --- 3. Concatenar ruido previo + trama
        tx_signal = np.concatenate([noise_idle, tx_frame])

        # --- 4. Canal selectivo en frecuencia (fading)
        tx_faded = np.convolve(tx_signal, h_freqsel, mode='same')

        # --- 5. Añadir ruido AWGN
        SNR   = 10**(snr_dB / 10)
        N0    = Es / SNR
        sigma = np.sqrt(N0/2)
        noise_sel = sigma * (np.random.randn(len(tx_faded)) + 1j*np.random.randn(len(tx_faded)))
        rx_signal_sel = tx_faded + noise_sel

        # --- 6. Detección de preámbulo
        x_detect = rx.detect(rx_signal_sel, 0, M, 8, M, zero_padding, up_ref, mag_threshold=None)

        if x_detect == -1:
            BER_tmp_sel[frame_idx] = 1.0
            SER_tmp_sel[frame_idx] = 1.0
            print("❌ No se detectó preámbulo FREQ-SEL")
            continue

        preamble_start = x_detect - (8 - 1) * M
        netid_len = 2 * M
        sfd_len = 2 * M + (M // 4)
        data_start = preamble_start + 8 * M + netid_len + sfd_len
        data_signal = rx_signal_sel[data_start : data_start + num_data_symbols * M]

        print(f"Preambulo detectado = {x_detect} Inicio estimado del preámbulo = {preamble_start}")

        # --- 7. Demodulación símbolo a símbolo
        symbols_rx = []
        for i in range(num_data_symbols):
            block = data_signal[i*M : (i+1)*M]
            if len(block) < M:
                break
            symbols_rx.append(rx.nTuple_former(block, M, B, T))
        symbols_rx = np.array(symbols_rx)
        bits_rx = rx.decode_symbols_to_bits(symbols_rx, SF)

        # --- 8. Cálculo de errores
        #num_symbol_errors = np.sum(symbols_tx[:num_data_symbols] != symbols_rx[:num_data_symbols])
        #num_bit_errors = np.sum(bits_tx[:num_data_symbols*SF] != bits_rx[:num_data_symbols*SF])

        #SER_tmp_sel[frame_idx] = num_symbol_errors / num_data_symbols
        #BER_tmp_sel[frame_idx] = num_bit_errors / (num_data_symbols * SF)

        # número de símbolos realmente demodulados
        num_proc_symbols = len(symbols_rx)

        if num_proc_symbols == 0:
            # no se demoduló nada: contar como fallo total
            SER_tmp_sel[frame_idx] = 1.0
            BER_tmp_sel[frame_idx] = 1.0
        else:
            # recortar el vector de símbolos transmitidos al número procesado
            tx_sym_chunk = symbols_tx[:num_proc_symbols]
            rx_sym_chunk = symbols_rx[:num_proc_symbols]

            # símbolo a símbolo
            num_symbol_errors = np.sum(tx_sym_chunk != rx_sym_chunk)
            SER_tmp_sel[frame_idx] = num_symbol_errors / num_proc_symbols

            # ahora bits: recortar según símbolos realmente procesados
            nbits_proc = num_proc_symbols * SF
            tx_bits_chunk = bits_tx[:nbits_proc]
            rx_bits_chunk = rx.decode_symbols_to_bits(rx_sym_chunk, SF)  # si ya tienes bits_rx, recorta directamente
            # asegúrate rx_bits_chunk tiene al menos nbits_proc (it should)
            rx_bits_chunk = rx_bits_chunk[:nbits_proc]

            num_bit_errors = np.sum(tx_bits_chunk != rx_bits_chunk)
            BER_tmp_sel[frame_idx] = num_bit_errors / nbits_proc

    # --- 9. Promedio sobre las N tramas
    SER_freqsel[idx] = np.mean(SER_tmp_sel)
    BER_freqsel[idx] = np.mean(BER_tmp_sel)

    print(f"SER promedio (Freq-sel): {SER_freqsel[idx]:.3e}, BER promedio (Freq-sel): {BER_freqsel[idx]:.3e}")
    
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