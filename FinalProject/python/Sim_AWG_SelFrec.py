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
zero_padding = 10                                    # Factor de zero padding en la FFT
num_data_symbols = num_bits // SF                   # Número de símbolos de datos

def lora_modulate(symbols_tx, M, B, T):
    preamble, netid, sfd = tx.preamble_netid_sfd(M, B, T)
    data_waveform = np.concatenate([tx.waveform_former(i, M, B, T) for i in symbols_tx])
    return np.concatenate([preamble, netid, sfd, data_waveform])

def compute_sigma_from_signal(tx_signal, snr_dB):
    """Calcula sigma de ruido complejo a partir de potencia real de tx_signal y SNR (dB)."""
    signal_power = np.mean(np.abs(tx_signal)**2)
    SNR_linear = 10**(snr_dB / 10.0)
    sigma = np.sqrt(signal_power / (2.0 * SNR_linear))
    return sigma

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

up_ref              = tx.waveform_former(0, M, B, T)
down_ref            = rx.make_down_ref(M, B, T)

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
        cfo_bins_frame = np.random.uniform(-1.0, 1.0)
        cfo_hz_frame = cfo_bins_frame * B / M

        # --- 1. Generar trama aleatoria
        bits_tx = np.random.randint(0, 2, size=num_bits)
        symbols_tx = tx.encode_bits_to_symbols(bits_tx, SF)
        tx_frame = lora_modulate(symbols_tx, M, B, T)

        jitter = np.random.randint(0, jitter_max_samples + 1)
        idle_len_frame = idle_base + jitter
        noise_idle = (np.random.randn(idle_len_frame) + 1j*np.random.randn(idle_len_frame)) / np.sqrt(2)
        noise_idle *= np.sqrt(np.mean(np.abs(tx_frame)**2))
        noise_idle *= 0.1

         # --- 3. Concatenar idle + trama + CFO
        tx_signal_sin_cfo = np.concatenate([noise_idle, tx_frame])
        tx_signal = tx.inject_cfo(tx_signal_sin_cfo, cfo_hz_frame, fs_eff=B)

        # --- 4. Aplicar canal y ruido AWGN
        sigma = compute_sigma_from_signal(tx_signal, snr_dB)
        noise = sigma * (np.random.randn(len(tx_signal)) + 1j*np.random.randn(len(tx_signal)))
        rx_signal = tx_signal + noise

        # --- 5. Detección de preámbulo
        x = rx.detect(rx_signal, 0, M, 8, M, zero_padding, up_ref, mag_threshold=None)
        if x != -1:
            preamble_start = x - (8 - 1) * M
            netid_len = 2 * M
            sfd_len = 2 * M + (M // 4)
            data_start_nom = preamble_start + 8 * M + netid_len + sfd_len
            #print(f"Preambulo detectado = {x}")
            #print(f"Inicio estimado del preámbulo = {preamble_start}")
            #print(f"Inicio de datos = {data_start_nom}, símbolos de datos a procesar = {num_symbols}")
        else:
            print("No se detectó preámbulo - saltando SNR")
            continue

        # --- 6. Sincronización fina
        x_sync, preamble_bin, preamble_bin_zp, cfo_hz = rx.sync(rx_signal, x, M, M, zero_padding, up_ref, down_ref, B)
        if x_sync == -1:
            print("Error en sincronización - saltando SNR")
            continue
        print(f"Sincronización exitosa: x_sync = {x_sync}, preamble_bin = {preamble_bin}, preamble_bin_zp = {preamble_bin_zp}, CFO = {cfo_hz} Hz")
             
        symbols_rx, num_avail = rx.demod_data(rx_signal, x_sync, num_symbols, M, zero_padding, up_ref, preamble_bin_zp)

        #--- 7. Cálculo de errores
        num_proc_symbols = num_avail
        if num_proc_symbols == 0:
            # no se demoduló nada: contar como fallo total
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

        cfo_bins_frame = np.random.uniform(-1.0, 1.0)
        cfo_hz_frame = cfo_bins_frame * B / M

        # --- 1. Generar trama aleatoria
        bits_tx = np.random.randint(0, 2, size=num_bits)
        symbols_tx = tx.encode_bits_to_symbols(bits_tx, SF)
        tx_frame = lora_modulate(symbols_tx, M, B, T)

        jitter = np.random.randint(0, jitter_max_samples + 1)
        idle_len_frame = idle_base + jitter
        noise_idle = (np.random.randn(idle_len_frame) + 1j*np.random.randn(idle_len_frame)) / np.sqrt(2)
        noise_idle *= np.sqrt(np.mean(np.abs(tx_frame)**2))
        noise_idle *= 0.1

        # --- 3. Concatenar ruido previo + trama
        tx_signal = np.concatenate([noise_idle, tx_frame])

        tx_faded_sin_cfo = np.convolve(tx_signal, h_freqsel, mode='same')
        tx_faded = tx.inject_cfo(tx_faded_sin_cfo, cfo_hz_frame, fs_eff=B)

        # --- 5. Añadir ruido AWGN
        sigma = compute_sigma_from_signal(tx_faded, snr_dB)
        noise_sel = sigma * (np.random.randn(len(tx_faded)) + 1j*np.random.randn(len(tx_faded)))
        rx_signal_sel = tx_faded + noise_sel

        # --- 6. Detección de preámbulo
        x = rx.detect(rx_signal_sel, 0, M, 8, M, zero_padding, up_ref, mag_threshold=None)
        if x != -1:
            preamble_start = x - (8 - 1) * M
            netid_len = 2 * M
            sfd_len = 2 * M + (M // 4)
            data_start_nom = preamble_start + 8 * M + netid_len + sfd_len
            #print(f"Preambulo detectado = {x}")
            #print(f"Inicio estimado del preámbulo = {preamble_start}")
            #print(f"Inicio de datos = {data_start_nom}, símbolos de datos a procesar = {num_symbols}")
        else:
            print("No se detectó preámbulo - saltando SNR")
            continue

        # --- 6. Sincronización fina
        x_sync, preamble_bin, preamble_bin_zp, cfo_hz = rx.sync(rx_signal_sel, x, M, M, zero_padding, up_ref, down_ref, B)
        if x_sync == -1:
            print("Error en sincronización - saltando SNR")
            continue
        print(f"Sincronización exitosa: x_sync = {x_sync}, preamble_bin = {preamble_bin}, preamble_bin_zp = {preamble_bin_zp}, CFO = {cfo_hz} Hz")

        symbols_rx, num_avail = rx.demod_data(rx_signal_sel, x_sync, num_symbols, M, zero_padding, up_ref, preamble_bin_zp)

        #--- 7. Cálculo de errores
        num_proc_symbols = num_avail

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