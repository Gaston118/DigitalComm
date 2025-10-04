import numpy as np
import matplotlib.pyplot as plt

# GENERACION DE LORA SYMBOLS.

SF = 7
M = 2**SF
B = 125e3         # Ancho de banda
T = 1/B           # Periodo de muestra
num_symbols = 500000
num_bits = num_symbols * SF

bits_tx = np.random.randint(0, 2, size=num_bits)

def encode_bits_to_symbols(bits, SF):
    n_sym = len(bits) // SF
    return np.array([
        sum(bits[i*SF + j] << (SF-1-j) for j in range(SF))
        for i in range(n_sym)
    ])

def waveform_former(symbol, M, B, T):
    k = np.arange(M)
    phase = ((symbol + k) % M) * (k * T * B) / M
    chirp_waveform = np.exp(1j * 2 * np.pi * phase) / np.sqrt(M)
    return chirp_waveform

symbols_tx = encode_bits_to_symbols(bits_tx, SF)

tx_signal = np.concatenate([waveform_former(i, M, B, T) for i in symbols_tx])

def nTuple_former(received_block, M, B, T):
    k = np.arange(M)
    down_chirp = np.exp(-1j * 2 * np.pi * (k * T * B) * k / M)
    reference_chirp = received_block * down_chirp
    spectrum = np.fft.fft(reference_chirp)
    return np.argmax(np.abs(spectrum))

symbols_rx = []
for idx in range(len(symbols_tx)):
    block = tx_signal[idx*M : (idx+1)*M]
    symbol_hat = nTuple_former(block, M, B, T)
    symbols_rx.append(symbol_hat)
symbols_rx = np.array(symbols_rx)

def decode_symbols_to_bits(symbols, SF):
    n_sym = len(symbols)
    return np.array([
        [(symbols[i] >> (SF-1-j)) & 1 for j in range(SF)]
        for i in range(n_sym)
    ]).flatten()

num_symbol_errors = np.sum(symbols_tx != symbols_rx)
SER_test = num_symbol_errors / num_symbols

bits_rx = decode_symbols_to_bits(symbols_rx, SF)

num_bits_errors = np.sum(bits_tx != bits_rx)
BER_test = num_bits_errors / num_bits

print("SF:", SF, "→ M =", M)
print("Bits transmitidos (primeros 20): ", bits_tx[:20])
print("Bits decodificados (primeros 20):", bits_rx[:20])
print("BER ideal: ", BER_test)
print("Símbolos transmitidos (primeros 10):", symbols_tx[:10])
print("Símbolos recibidos    (primeros 10):", symbols_rx[:10])
print("SER ideal: ", SER_test)

snr_dB_range = np.arange(-11, 1, 1)                        # Es/N0 (dB)
EsN0_dB_range = snr_dB_range + 10*np.log10(M)              # Para simular se suma, ya que SNR dB = Es/N0 - 10log10(M)
Es = 1                                                     # Energía por símbolo (normalizada)
BER_awgn = np.zeros_like(snr_dB_range, dtype=float)
SER_awgn = np.zeros_like(snr_dB_range, dtype=float)

h_freqsel    = np.array([np.sqrt(0.8), np.sqrt(0.2)])      # √0.8 δ[n] + √0.2 δ[n-1]
BER_freqsel  = np.zeros_like(snr_dB_range, dtype=float)
SER_freqsel  = np.zeros_like(snr_dB_range, dtype=float)

Es = 1.0        # Energía media por símbolo (señal normalizada)
Eb = Es / SF    # Energía por bit

for idx, snr_dB in enumerate(EsN0_dB_range):
    SNR   = 10**(snr_dB / 10)               # relación lineal Potencia_señal / Potencia_ruido
    N0    = Es / SNR                        # densidad espectral de ruido
    sigma = np.sqrt(N0/2)                   # desviación típica por dimensión / estandar

    # Generación de ruido AWGN complejo
    noise = sigma * (np.random.randn(len(tx_signal)) + 1j*np.random.randn(len(tx_signal)))

    # Señal recibida con AWGN
    rx_signal = tx_signal + noise

    # Demodulación símbolo a símbolo
    symbols_rx = []
    for sym_idx in range(len(symbols_tx)):
        block = rx_signal[sym_idx*M : (sym_idx+1)*M]
        symbols_rx.append(nTuple_former(block, M, B, T))
    symbols_rx = np.array(symbols_rx)

    # Decodificación a bits
    bits_rx_awgn = decode_symbols_to_bits(symbols_rx, SF)

    # Cálculo de tasas de error
    SER_awgn[idx] = np.sum(symbols_tx != symbols_rx) / num_symbols
    BER_awgn[idx] = np.sum(bits_tx != bits_rx_awgn) / num_bits

    # Se realiza ahora para la señal selectiva en frecuencia
    tx_faded = np.convolve(tx_signal, h_freqsel, mode='full')[:len(tx_signal)]

    # AWGN con el mismo sigma ya calculado
    noise_sel     = sigma * (np.random.randn(len(tx_faded)) + 1j*np.random.randn(len(tx_faded)))
    rx_signal_sel = tx_faded + noise_sel

    # Demodulación símbolo a símbolo (idéntica a la de AWGN)
    symbols_rx_sel = []
    for sym_idx in range(len(symbols_tx)):
        block_sel = rx_signal_sel[sym_idx*M : (sym_idx+1)*M]
        symbols_rx_sel.append(nTuple_former(block_sel, M, B, T))
    symbols_rx_sel = np.array(symbols_rx_sel)

    # Decodificación a bits
    bits_rx_sel = decode_symbols_to_bits(symbols_rx_sel, SF)

    # Tasas de error
    SER_freqsel[idx] = np.sum(symbols_tx != symbols_rx_sel) / num_symbols
    BER_freqsel[idx] = np.sum(bits_tx != bits_rx_sel) / num_bits

    print(f"SNRdB = {snr_dB_range[idx]:5.1f}: "
          f"BER_AWGN = {BER_awgn[idx]:.3e}, SER_AWGN = {SER_awgn[idx]:.3e} | "
          f"BER_FreqSel = {BER_freqsel[idx]:.3e}, SER_FreqSel = {SER_freqsel[idx]:.3e}")
    
plt.figure()
plt.semilogy(snr_dB_range, BER_awgn,    'o-',  label='BER Flat FSCM')
plt.semilogy(snr_dB_range, SER_awgn,    's-',  label='SER Flat FSCM')
plt.semilogy(snr_dB_range, BER_freqsel, 'o--', label='BER Freq-sel FSCM')
plt.semilogy(snr_dB_range, SER_freqsel, 's--', label='SER Freq-sel FSCM')

plt.xlim(-12, 0)
plt.ylim(1e-5, 1e-1)
plt.xticks(np.arange(-12, 1, 1))
plt.grid(True, which='both', ls='--', lw=0.5)
plt.xlabel('SNR (dB)')
plt.ylabel('BER / SER')
plt.title('Uncoded BER / SER – Flat vs Freq-sel FSCM (SF = 7)')
plt.legend()
plt.show()