import numpy as np

#############################################################################################
#                                                                                           #
#  ████████ ██████   █████  ███    ██ ███████ ███    ███ ██ ███████ ██  ██████  ███    ██   #
#     ██    ██   ██ ██   ██ ████   ██ ██      ████  ████ ██ ██      ██ ██    ██ ████   ██   #
#     ██    ██████  ███████ ██ ██  ██ ███████ ██ ████ ██ ██ ███████ ██ ██    ██ ██ ██  ██   #
#     ██    ██   ██ ██   ██ ██  ██ ██      ██ ██  ██  ██ ██      ██ ██ ██    ██ ██  ██ ██   #
#     ██    ██   ██ ██   ██ ██   ████ ███████ ██      ██ ██ ███████ ██  ██████  ██   ████   #
#                                                                                           #
#############################################################################################

#===========================================================================================
#                               CODIFICACIÓN DE BITS A SÍMBOLOS
#===========================================================================================
def encode_bits_to_symbols(bits, SF):
    n_sym = len(bits) // SF
    return np.array([
        sum(bits[i*SF + j] << (SF-1-j) for j in range(SF))
        for i in range(n_sym)
    ])

#===========================================================================================
#                               FORMACIÓN DE LA FORMA DE ONDA
#===========================================================================================
def waveform_former(symbol, M, B, T):
    k = np.arange(M)
    phase = ((symbol + k) % M) * (k * T * B) / M
    chirp_waveform = np.exp(1j * 2 * np.pi * phase) / np.sqrt(M)
    return chirp_waveform

#===========================================================================================
#                           GENERACIÓN DE PREÁMBULO, NETID Y SFD
#===========================================================================================
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

#===========================================================================================
#                          FUNCIONES DE SIMULACIÓN DE CFO
#===========================================================================================
def inject_cfo(x: np.ndarray, cfo_hz: float, fs_eff: float, start_idx: int = 0) -> np.ndarray:
    if cfo_hz == 0.0:
        return x
    y = np.array(x, copy=True)
    n = np.arange(len(y) - start_idx, dtype=float)
    rot = np.exp(1j * 2 * np.pi * cfo_hz * n / float(fs_eff))
    y[start_idx:] *= rot
    return y