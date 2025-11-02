import numpy as np
import matplotlib.pyplot as plt

#===========================================================================================
#                        VISUALIZACIÓN DE LA TRAMA EN EL TIEMPO
#===========================================================================================

def plot_lora_frame(signal, M, SF, B, preamble_len=8, netid_len=2):
    """
    Grafica la trama LoRa completa mostrando sus diferentes secciones
    """
    # Configuración de tiempos
    T = 1/B
    t = np.arange(len(signal)) * T * 1e3  # Tiempo en milisegundos
    
    # Límites de cada sección
    preamble_end = preamble_len * M
    netid_end = preamble_end + netid_len * M
    sfd_end = netid_end + int(2.25 * M)
    
    # Crear figura con subplots
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    
    # 1. Parte Real de toda la señal
    axes[0].plot(t, np.real(signal), linewidth=0.5)
    axes[0].axvline(t[preamble_end], color='r', linestyle='--', label='Fin Preámbulo')
    axes[0].axvline(t[netid_end], color='g', linestyle='--', label='Fin NetID')
    axes[0].axvline(t[sfd_end], color='b', linestyle='--', label='Fin SFD')
    axes[0].set_ylabel('Amplitud')
    axes[0].set_title(f'Trama LoRa Completa (SF={SF})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Parte Imaginaria de toda la señal
    axes[1].plot(t, np.imag(signal), linewidth=0.5, color='orange')
    axes[1].axvline(t[preamble_end], color='r', linestyle='--')
    axes[1].axvline(t[netid_end], color='g', linestyle='--')
    axes[1].axvline(t[sfd_end], color='b', linestyle='--')
    axes[1].set_ylabel('Amplitud')
    axes[1].set_title('Parte Imaginaria')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Magnitud
    axes[2].plot(t, np.abs(signal), linewidth=0.5, color='purple')
    axes[2].axvline(t[preamble_end], color='r', linestyle='--')
    axes[2].axvline(t[netid_end], color='g', linestyle='--')
    axes[2].axvline(t[sfd_end], color='b', linestyle='--')
    axes[2].set_ylabel('Magnitud')
    axes[2].set_title('Magnitud de la Señal')
    axes[2].grid(True, alpha=0.3)
    
    # 4. Fase instantánea
    phase = np.unwrap(np.angle(signal))
    axes[3].plot(t, phase, linewidth=0.5, color='green')
    axes[3].axvline(t[preamble_end], color='r', linestyle='--')
    axes[3].axvline(t[netid_end], color='g', linestyle='--')
    axes[3].axvline(t[sfd_end], color='b', linestyle='--')
    axes[3].set_ylabel('Fase (rad)')
    axes[3].set_xlabel('Tiempo (ms)')
    axes[3].set_title('Fase Instantánea')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_spectrogram(signal, B, SF, M, section='preamble'):
    """
    Espectrograma con zoom en una sección específica
    """
    from scipy import signal as scipy_signal
    
    fs = 2 * B
    
    # Definir región de interés
    if section == 'preamble':
        start_sample = 0
        end_sample = 8 * M
        title = "Preambulo (8 Up-Chirps)"
    elif section == 'sfd':
        start_sample = 10 * M
        end_sample = start_sample + int(2.25 * M)
        title = "SFD (Down-Chirps)"
    elif section == 'data':
        start_sample = int(12.25 * M)
        end_sample = start_sample + 10 * M
        title = "Primeros 10 Simbolos de Datos"
    else:
        start_sample = 0
        end_sample = len(signal)
        title = "Senal Completa"
    
    signal_zoom = signal[start_sample:end_sample]
    
    nperseg = M // 2
    noverlap = int(nperseg * 0.95)
    
    f, t, Sxx = scipy_signal.spectrogram(
        signal_zoom, 
        fs=fs, 
        nperseg=nperseg,
        noverlap=noverlap,
        window='hann',
        scaling='spectrum',
        return_onesided=False,
        mode='magnitude'
    )
    
    # Reordenar frecuencias
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    
    Sxx_dB = 10 * np.log10(np.abs(Sxx) + 1e-12)
    
    plt.figure(figsize=(14, 7))
    plt.pcolormesh(t * 1e3, f/1e3, Sxx_dB, 
                   shading='gouraud', 
                   cmap='jet',
                   vmin=np.max(Sxx_dB) - 50,
                   vmax=np.max(Sxx_dB))
    
    plt.colorbar(label='Potencia (dB)')
    plt.ylabel('Frecuencia (kHz)')
    plt.xlabel('Tiempo (ms)')
    plt.title(f'Espectrograma LoRa - {title} (SF={SF})')
    plt.ylim([-B/1e3, B/1e3])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()