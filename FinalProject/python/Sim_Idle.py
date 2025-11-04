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

cfo_bins_frame = np.random.uniform(-2.0, 2.0)
cfo_hz_frame = cfo_bins_frame * B / M

print(f"Generando trama LoRa con CFO de {cfo_hz_frame:.2f} Hz ({cfo_bins_frame:.2f} bins)...")

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
esp.plot_lora_frame(tx_signal, M, SF, B)
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
zero_padding    = 10

#===========================================================================
#                                DETECCIÓN 
#===========================================================================
x = rx.detect(tx_signal, 0, M, 8, M, zero_padding, up_ref, mag_threshold=None)
if x != -1:
    preamble_start = x - (8 - 1) * M
    netid_len = 2 * M
    sfd_len = 2 * M + (M // 4)
    data_start_nom = preamble_start + 8 * M + netid_len + sfd_len
    print(f"Preambulo detectado = {x}")
    print(f"Inicio estimado del preámbulo = {preamble_start}")
    print(f"Inicio de datos = {data_start_nom}, símbolos de datos a procesar = {num_symbols}")
else:
    raise RuntimeError("No se detectó preámbulo")

#===========================================================================
#                                SINCRONIZACIÓN
#===========================================================================
x_sync, preamble_bin, preamble_bin_zp, cfo_hz = rx.sync(tx_signal, x, M, M, zero_padding, up_ref, down_ref, B)
if x_sync == -1:
    raise RuntimeError("Error en sincronización")
print(f"Sincronización exitosa: x_sync = {x_sync}, preamble_bin = {preamble_bin}, preamble_bin_zp = {preamble_bin_zp}, CFO = {cfo_hz} Hz")

cfo_bins_inj = cfo_hz_frame * M / B
cfo_bins_est_cont = (cfo_hz or 0.0) * M / B
print(f"CFO inj(cont)={cfo_bins_inj:.2f} bins | CFO est(cont)={cfo_bins_est_cont:.2f} bins | preamble_bin(entero)={(preamble_bin if preamble_bin<=M//2 else preamble_bin-M):.0f} bins")

def wrap_frac(x): return ((x + 0.5) % 1.0) - 0.5
pre_signed = preamble_bin if preamble_bin <= M//2 else preamble_bin - M
print(f"[CFO] entero(ref)={pre_signed:+.0f} | frac inj={wrap_frac(cfo_bins_inj-pre_signed):+.3f} | frac est={wrap_frac(cfo_bins_est_cont-pre_signed):+.3f}")

cfo_bins_est_zp = preamble_bin_zp / float(zero_padding)
print(f"[ZP] preamble_bin_zp={preamble_bin_zp} -> CFO_bins_est_ZP={cfo_bins_est_zp:.3f} (mod 1 bin)")

symbols_rx, num_avail = rx.demod_data(tx_signal, x_sync, num_symbols, M, zero_padding, up_ref, preamble_bin_zp)

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