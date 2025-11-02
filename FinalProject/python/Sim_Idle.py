import numpy as np

import Espectrograma as esp
import Transimicion as tx
import Recepcion as rx

#############################################################################################
#                                                                                           #
#  ████████ ██████   █████  ███    ██ ███████ ███    ███ ██ ███████ ██  ██████  ███    ██   #
#     ██    ██   ██ ██   ██ ████   ██ ██      ████  ████ ██ ██      ██ ██    ██ ████   ██   #
#     ██    ██████  ███████ ██ ██  ██ ███████ ██ ████ ██ ██ ███████ ██ ██    ██ ██ ██  ██   #
#     ██    ██   ██ ██   ██ ██  ██ ██      ██ ██  ██  ██ ██      ██ ██ ██    ██ ██  ██ ██   #
#     ██    ██   ██ ██   ██ ██   ████ ███████ ██      ██ ██ ███████ ██  ██████  ██   ████   #
#                                                                                           #
#############################################################################################

SF = 7                                              # Spreading Factor
M = 2**SF                                           # Número de bits por símbolo
B = 125e3                                           # Ancho de banda
T = 1/B                                             # Periodo de muestra
num_symbols = 2000                                  # Número de símbolos a transmitir
num_bits = num_symbols * SF                         # Número de bits a transmitir
bits_tx = np.random.randint(0, 2, size=num_bits)    # Bits a transmitir

#===========================================================================================
#                               GENERACION DE LA TRAMA LORA
#===========================================================================================
symbols_tx = tx.encode_bits_to_symbols(bits_tx, SF)

while True:
    cfo_bins_frame = np.random.uniform(-3.0, 3.0)
    if abs(cfo_bins_frame) >= 0.5:   
        break
cfo_hz_frame = cfo_bins_frame * B / M

def lora_modulate(symbols_tx, M, B, T):
    preamble, netid, sfd = tx.preamble_netid_sfd(M, B, T)
    data_waveform = np.concatenate([tx.waveform_former(i, M, B, T) for i in symbols_tx])
    return np.concatenate([preamble, netid, sfd, data_waveform])

tx_signal_sin_cfo = lora_modulate(symbols_tx, M, B, T)

tx_signal = tx.inject_cfo(tx_signal_sin_cfo, cfo_hz_frame, fs_eff=B)

# ===========================================================================================
#                          GENERACIÓN DE GRÁFICAS DE LA TRAMA LORA
# ===========================================================================================

#print("\nGenerando gráficas de la trama LoRa...")
# 1. Gráfica completa de la trama
#esp.plot_lora_frame(tx_signal, M, SF, B)
# 2. Zoom en preámbulo
#esp.plot_spectrogram(tx_signal, B, SF, M, section='preamble')
# 3. Zoom en SFD
#esp.plot_spectrogram(tx_signal, B, SF, M, section='sfd')
# 4. Zoom en datos
#esp.plot_spectrogram(tx_signal, B, SF, M, section='data')

###########################################################################
#                                                                         #
#  ██████  ███████  ██████ ███████ ██████   ██████ ██  ██████  ███    ██  #
#  ██   ██ ██      ██      ██      ██   ██ ██      ██ ██    ██ ████   ██  #
#  ██████  █████   ██      █████   ██████  ██      ██ ██    ██ ██ ██  ██  #
#  ██   ██ ██      ██      ██      ██      ██      ██ ██    ██ ██  ██ ██  #
#  ██   ██ ███████  ██████ ███████ ██       ██████ ██  ██████  ██   ████  #
#                                                                         #
###########################################################################

up_ref          = tx.waveform_former(0, M, B, T)
down_ref        = rx.make_down_ref(M, B, T)
zero_padding    = 1

x = rx.detect(tx_signal, 0, M, 8, M, zero_padding, up_ref, mag_threshold=None)

if x != -1:
    preamble_start = x - (8 - 1) * M
    netid_len = 2 * M
    sfd_len = 2 * M + (M // 4)  
    data_start = preamble_start + 8 * M + netid_len + sfd_len

    num_data_symbols = num_symbols 

    print(f"Preambulo detectado = {x}")
    print(f"Inicio estimado del preámbulo = {preamble_start}")
    print(f"Inicio de datos = {data_start}, símbolos de datos a procesar = {num_data_symbols}")
else:
    print("No se detectó preámbulo")

x_sync, preamble_bin, cfo_hz = rx.sync(tx_signal, x, M, M, zero_padding, up_ref, down_ref, B)

if x_sync == -1:
    print("Error en sincronización")
else:
    print(f"Sincronización exitosa: x_sync = {x_sync}, preamble_bin = {preamble_bin}, CFO = {cfo_hz} Hz")

data_start = x_sync

expected_data_start = int(round(12.25 * M)) 
print(f"Check data start: esperado≈{expected_data_start}, x_sync={x_sync}, delta={x_sync-expected_data_start}")
print(f"CFO inj={cfo_hz_frame * M / B:.2f} bins, est={ (preamble_bin if preamble_bin<=M//2 else preamble_bin-M):.2f} bins")

#===========================================================================================
#                  Extraer los datos desde la posición detectada
#===========================================================================================

# Extraer ventana de datos desde data_start
data_signal = tx_signal[data_start : data_start + num_data_symbols * M]

# Número de símbolos completos realmente disponibles
num_avail = len(data_signal) // M

# Demodulación usando solo símbolos completos
symbols_rx = []
for i in range(num_avail):
    block = data_signal[i*M : (i+1)*M]
    raw_bin = rx.nTuple_former(block, M, B, T)          # 0..M-1
    sym_hat = (int(raw_bin) - int(preamble_bin)) % M    # eliminar CFO en bin
    symbols_rx.append(sym_hat)
symbols_rx = np.array(symbols_rx, dtype=int)

# Métricas usando la cantidad procesada
tx_sym_chunk = symbols_tx[:num_avail]
rx_sym_chunk = symbols_rx[:num_avail]
num_symbol_errors = np.sum(tx_sym_chunk != rx_sym_chunk)
SER = num_symbol_errors / max(1, num_avail)

# Bits
nbits_proc = num_avail * SF
tx_bits_chunk = bits_tx[:nbits_proc]
rx_bits_chunk = rx.decode_symbols_to_bits(rx_sym_chunk, SF)[:nbits_proc]
num_bit_errors = np.sum(tx_bits_chunk != rx_bits_chunk)
BER = num_bit_errors / max(1, nbits_proc)

print("\n================= RESULTADOS =================")
print(f"SF = {SF}, M = {M}")
print("Bits TX (primeros 20): ", bits_tx[:20])
print("Bits RX (primeros 20):", rx_bits_chunk[:20])
print("Símbolos TX (primeros 10):", symbols_tx[:10])
print("Símbolos RX (primeros 10):", symbols_rx[:10])
print(f"SER ideal: {SER}")
print(f"BER ideal: {BER}")