import numpy as np
import matplotlib.pyplot as plt

SF = 7
M = 2**SF
B = 125e3         # Ancho de banda
T = 1/B           # Periodo de muestra
num_symbols = 20000
num_bits = num_symbols * SF

bits_tx = np.random.randint(0, 2, size=num_bits)

encoder = np.array([
    sum(bits_tx[i*SF + j] << (SF-1-j) for j in range(SF))
    for i in range(num_symbols)
])

def waveform_former(symbol, M, B, T):
    k = np.arange(M)
    phase = ((symbol + k) % M) * (k * T * B) / M
    chirp_waveform = np.exp(1j * 2 * np.pi * phase) / np.sqrt(M)
    return chirp_waveform

tx_signal = np.concatenate([waveform_former(i, M, B, T) for i in encoder])

def nTuple_former(received_block, M, B, T):
    k = np.arange(M)
    down_chirp = np.exp(-1j * 2 * np.pi * (k * T * B) * k / M)
    reference_chirp = received_block * down_chirp
    spectrum = np.fft.fft(reference_chirp)
    return np.argmax(np.abs(spectrum))

symbols_rx = []
for idx in range(len(encoder)):
    block = tx_signal[idx*M : (idx+1)*M]
    symbol_hat = nTuple_former(block, M, B, T)
    symbols_rx.append(symbol_hat)
symbols_rx = np.array(symbols_rx)

decoder = np.array([
    [(symbols_rx[i] >> (SF-1-j)) & 1 for j in range(SF)]
    for i in range(len(symbols_rx))
]).flatten()

num_symbol_errors = np.sum(encoder != symbols_rx)
SER_test = num_symbol_errors / num_symbols

num_errors = np.sum(bits_tx != decoder)
BER_test = num_errors / num_bits

print("=" * 70)
print(f"{'CONFIGURACIÓN DE SIMULACIÓN LoRa':^70}")
print("=" * 70)
print(f"SF: {SF} → M = 2^{SF} = {M} símbolos")
print(f"Ancho de banda: {B/1000:.0f} kHz")
print(f"Número de símbolos: {num_symbols:,}")
print(f"Número de bits: {num_bits:,}")
print()

print("TRANSMISIÓN Y RECEPCIÓN IDEAL (Sin ruido)")
print("-" * 50)
print(f"Bits TX  (20 primeros): {' '.join(map(str, bits_tx[:20]))}")
print(f"Bits RX  (20 primeros): {' '.join(map(str, decoder[:20]))}")
print()
print(f"Símbolos TX (10 primeros): {encoder[:10]}")
print(f"Símbolos RX (10 primeros): {symbols_rx[:10]}")
print()
print(f" BER ideal: {BER_test:.6f}")
print(f" SER ideal: {SER_test:.6f}")
print()

print(" INICIANDO SIMULACIÓN CON RUIDO...")
print("=" * 70)

snr_dB_range = np.arange(10, 30, 2)   # Es/N0 (dB)
EbN0_dB_range = snr_dB_range - 10*np.log10(M)   # Eb/N0 (dB)
BER_awgn = np.zeros_like(snr_dB_range, dtype=float)
SER_awgn = np.zeros_like(snr_dB_range, dtype=float)

h_freqsel    = np.array([np.sqrt(0.8), np.sqrt(0.2)])      # √0.8 δ[n] + √0.2 δ[n-1]
BER_freqsel  = np.zeros_like(snr_dB_range, dtype=float)
SER_freqsel  = np.zeros_like(snr_dB_range, dtype=float)

Es = 1.0        # Energía media por símbolo (señal normalizada)
Eb = Es / SF    # Energía por bit

for idx, EbN0_dB in enumerate(EbN0_dB_range):
    EbN0 = 10**(EbN0_dB / 10)       # Eb/N0 lineal
    EsN0 = EbN0 * SF                # Es/N0 lineal
    N0    = Es / EsN0               # densidad de ruido
    sigma = np.sqrt(N0/2)           # desviación típica por dimensión

    # Generación de ruido AWGN complejo
    noise = sigma * (np.random.randn(len(tx_signal)) + 1j*np.random.randn(len(tx_signal)))

    # Señal recibida con AWGN
    rx_signal = tx_signal + noise

    # Demodulación símbolo a símbolo
    symbols_rx = []
    for sym_idx in range(len(encoder)):
        block = rx_signal[sym_idx*M : (sym_idx+1)*M]
        symbols_rx.append(nTuple_former(block, M, B, T))
    symbols_rx = np.array(symbols_rx)

    # Decodificación a bits
    decoder = np.array([
        [(symbols_rx[i] >> (SF-1-j)) & 1 for j in range(SF)]
        for i in range(len(symbols_rx))
    ]).flatten()

    # Cálculo de tasas de error
    SER_awgn[idx] = np.sum(encoder != symbols_rx) / num_symbols
    BER_awgn[idx] = np.sum(bits_tx != decoder) / num_bits

    # Realizamos lo mismo pero ahora con la senal desplazada en frecuencia
    tx_faded = np.convolve(tx_signal, h_freqsel, mode='full')[:len(tx_signal)]

    # AWGN con la misma sigma que ya calculaste
    noise_sel     = sigma * (np.random.randn(len(tx_faded)) + 1j*np.random.randn(len(tx_faded)))
    rx_signal_sel = tx_faded + noise_sel

    # Demodulación símbolo a símbolo (idéntica a la de AWGN)
    symbols_rx_sel = []
    for sym_idx in range(len(encoder)):
        block_sel = rx_signal_sel[sym_idx*M : (sym_idx+1)*M]
        symbols_rx_sel.append(nTuple_former(block_sel, M, B, T))
    symbols_rx_sel = np.array(symbols_rx_sel)

    # Decodificación a bits
    decoder_sel = np.array([
        [(symbols_rx_sel[i] >> (SF-1-j)) & 1 for j in range(SF)]
        for i in range(len(symbols_rx_sel))
    ]).flatten()

    # Tasas de error
    SER_freqsel[idx] = np.sum(encoder != symbols_rx_sel) / num_symbols
    BER_freqsel[idx] = np.sum(bits_tx != decoder_sel) / num_bits

    # Resultados formateados
    print(f" Eb/N0 = {EbN0_dB:+5.1f} dB │ "
          f"🔵 AWGN: BER={BER_awgn[idx]:.2e} SER={SER_awgn[idx]:.2e} │ "
          f"🔴 FreqSel: BER={BER_freqsel[idx]:.2e} SER={SER_freqsel[idx]:.2e}")

# AQUÍ ES DONDE DEBE IR plt.figure() - FUERA DEL BUCLE
print()
print("=" * 70)
print("📈 Generando gráfico de resultados...")
plt.figure()

plt.semilogy(EbN0_dB_range, BER_awgn,    'o-', label='BER (Flat)')
plt.semilogy(EbN0_dB_range, SER_awgn,    's-', label='SER (Flat)')
plt.semilogy(EbN0_dB_range, BER_freqsel, 'o--',label='BER (Freq Sel)')
plt.semilogy(EbN0_dB_range, SER_freqsel, 's--',label='SER (Freq Sel)')

plt.ylim(1e-4, 1)

plt.xlim(EbN0_dB_range[0], EbN0_dB_range[-1])   # -9 … +11 dB

plt.grid(True, which='both', ls='--', lw=0.5)
plt.xlabel('SNR = Eb/N0 (dB)')
plt.ylabel('Tasa de error')
plt.title('Curvas de BER y SER – Flat vs Freq Sel FSCM')
plt.legend()
plt.show()