import numpy as np
import Transimicion as tx
import Recepcion as rx
from datetime import datetime

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
rf_freq = 470e6                                     # Frecuencia portadora en Hz
num_symbols = 2000                                  # Número de símbolos por trama
num_bits = num_symbols * SF                         # Número de bits por trama
num_frames = 10                                     # Número de tramas a generar
silence_duration = 1000                             # Duración del silencio entre tramas (muestras)
zero_padding = 10                                   # Factor de zero padding en la FFT

# Configuración de archivo de salida
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"01_simulacion_lora_{timestamp}.txt"

#===========================================================================================
#                       GENERACIÓN DE MÚLTIPLES TRAMAS LORA
#===========================================================================================

def generate_lora_frame(symbols_tx, M, B, T, cfo_hz_frame):
    """Genera una trama LoRa completa con CFO"""
    preamble, netid, sfd = tx.preamble_netid_sfd(M, B, T)
    data_waveform = np.concatenate([tx.waveform_former(i, M, B, T) for i in symbols_tx])
    tx_signal_sin_cfo = np.concatenate([preamble, netid, sfd, data_waveform])
    return tx.inject_cfo(tx_signal_sin_cfo, cfo_hz_frame, fs_eff=B)

def generate_multiple_frames(num_frames, num_symbols, SF, M, B, T, silence_duration, log_file):
    """Genera múltiples tramas LoRa con silencios entre ellas"""
    all_frames = []
    frame_info = []  # Para almacenar información de cada trama
    current_sample = 0
    
    for frame_idx in range(num_frames):
        # Generar bits y símbolos para esta trama
        bits_tx = np.random.randint(0, 2, size=num_symbols * SF)
        symbols_tx = tx.encode_bits_to_symbols(bits_tx, SF)
        
        # Generar CFO único para esta trama
        cfo_bins_frame = np.random.uniform(-2.0, 2.0)
        cfo_hz_frame = cfo_bins_frame * B / M
        
        # Generar la trama LoRa
        frame_signal = generate_lora_frame(symbols_tx, M, B, T, cfo_hz_frame)
        
        # Agregar silencio antes de la trama (excepto para la primera)
        if frame_idx > 0:
            silence = np.zeros(silence_duration, dtype=complex)
            all_frames.append(silence)
            current_sample += silence_duration
        
        # Almacenar información de la trama
        frame_info.append({
            'frame_idx': frame_idx,
            'bits_tx': bits_tx,
            'symbols_tx': symbols_tx,
            'cfo_hz_frame': cfo_hz_frame,
            'cfo_bins_frame': cfo_bins_frame,
            'frame_start_idx': current_sample,
            'frame_length': len(frame_signal)
        })
        
        # Agregar la trama
        all_frames.append(frame_signal)
        
        msg = (f"Trama {frame_idx+1}/{num_frames}: CFO = {cfo_hz_frame:.2f} Hz ({cfo_bins_frame:.2f} bins), "
               f"Inicio = {current_sample}, Longitud = {len(frame_signal)} muestras\n")
        log_file.write(msg)
        
        current_sample += len(frame_signal)
    
    # Convertir a array numpy
    combined_signal = np.concatenate(all_frames)
    
    summary = (f"\nSeñal combinada generada:\n"
               f"- Número de tramas: {num_frames}\n"
               f"- Duración de silencio entre tramas: {silence_duration} muestras\n"
               f"- Longitud total de la señal: {len(combined_signal)} muestras ({len(combined_signal)/M:.1f} símbolos)\n")
    log_file.write(summary)
    
    return combined_signal, frame_info

# Abrir archivo de log y mantenerlo abierto durante toda la ejecución
log_file = open(output_file, 'w', encoding='utf-8')

try:
    log_file.write("="*80 + "\n")
    log_file.write(f"SIMULACIÓN DE {num_frames} TRAMAS LoRa EN CANAL IDEAL\n")
    log_file.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write("="*80 + "\n\n")
    
    log_file.write("--- PARÁMETROS DE SIMULACIÓN ---\n")
    log_file.write(f"SF = {SF}\n")
    log_file.write(f"M = {M}\n")
    log_file.write(f"B = {B/1e3:.0f} kHz\n")
    log_file.write(f"RF Freq = {rf_freq/1e6:.0f} MHz\n")
    log_file.write(f"Símbolos por trama = {num_symbols}\n")
    log_file.write(f"Bits por trama = {num_bits}\n")
    log_file.write(f"Número de tramas = {num_frames}\n")
    log_file.write(f"Silencio entre tramas = {silence_duration} muestras\n")
    log_file.write(f"Zero padding = {zero_padding}\n\n")
    
    log_file.write("="*80 + "\n")
    log_file.write("GENERACIÓN DE TRAMAS\n")
    log_file.write("="*80 + "\n\n")

    # Generar todas las tramas
    tx_signal, frames_info = generate_multiple_frames(
        num_frames, num_symbols, SF, M, B, T, silence_duration, log_file
    )

    ###########################################################################
    #                                                                         #
    #  ██████  ███████  ██████ ███████ ██████   ██████ ██  ██████  ███    ██  #
    #  ██   ██ ██      ██      ██      ██   ██ ██      ██ ██    ██ ████   ██  #
    #  ██████  █████   ██      █████   ██████  ██      ██ ██    ██ ██ ██  ██  #
    #  ██   ██ ██      ██      ██      ██      ██      ██ ██    ██ ██  ██ ██  #
    #  ██   ██ ███████  ██████ ███████ ██       ██████ ██  ██████  ██   ████  #
    #                                                                         #
    ###########################################################################

    ###########################################################################
    #                                                                         #
    #  ██████  ███████  ██████ ███████ ██████   ██████ ██  ██████  ███    ██  #
    #  ██   ██ ██      ██      ██      ██   ██ ██      ██ ██    ██ ████   ██  #
    #  ██████  █████   ██      █████   ██████  ██      ██ ██    ██ ██ ██  ██  #
    #  ██   ██ ██      ██      ██      ██      ██      ██ ██    ██ ██  ██ ██  #
    #  ██   ██ ███████  ██████ ███████ ██       ██████ ██  ██████  ██   ████  #
    #                                                                         #
    ###########################################################################

    up_ref = tx.waveform_former(0, M, B, T)
    down_ref = rx.make_down_ref(M, B, T)

    #===========================================================================
    #                       PROCESAMIENTO DE MÚLTIPLES TRAMAS
    #===========================================================================

    def process_multiple_frames(tx_signal, frames_info, M, zero_padding, up_ref, down_ref, B, num_symbols, SF, rf_freq, log_file):
        """Procesa y demodula múltiples tramas LoRa"""
        
        all_results = []
        current_pos = 0
        
        for frame_info in frames_info:
            frame_idx = frame_info['frame_idx']
            expected_start = frame_info['frame_start_idx']
            
            log_file.write("\n" + "="*80 + "\n")
            log_file.write(f"PROCESANDO TRAMA {frame_idx+1}/{len(frames_info)}\n")
            log_file.write("="*80 + "\n\n")
            
            #===========================================================================
            #                                DETECCIÓN 
            #===========================================================================
            x = rx.detect(tx_signal[current_pos:], 0, M, 8, M, zero_padding, up_ref, mag_threshold=None)
            
            if x != -1:
                # Ajustar la posición al inicio real en la señal completa
                actual_start = current_pos + x - (8 - 1) * M
                preamble_start = x - (8 - 1) * M
                
                netid_len = 2 * M
                sfd_len = 2 * M + (M // 4)
                data_start_nom = preamble_start + 8 * M + netid_len + sfd_len
                
                log_file.write("✓ Preámbulo detectado:\n")
                log_file.write(f"  - Posición relativa: {x}\n")
                log_file.write(f"  - Inicio preámbulo (relativo): {preamble_start}\n")
                log_file.write(f"  - Inicio datos (relativo): {data_start_nom}\n")
                log_file.write(f"  - Posición real en señal: {actual_start}\n")
                log_file.write(f"  - Posición esperada: {expected_start}\n")
                log_file.write(f"  - Diferencia: {actual_start - expected_start} muestras\n\n")
            else:
                log_file.write(f"⚠ ERROR: No se detectó preámbulo en la trama {frame_idx+1}\n\n")
                all_results.append({
                    'frame_idx': frame_idx,
                    'status': 'DETECTION_FAILED',
                    'error': 'No se detectó preámbulo'
                })
                # Avanzar posición basándonos en la longitud esperada de la trama
                current_pos = expected_start + frame_info['frame_length']
                continue
            
            #===========================================================================
            #                                SINCRONIZACIÓN
            #===========================================================================
            x_sync, preamble_bin, preamble_bin_zp, cfo_hz = rx.sync(
                tx_signal[current_pos:], x, M, M, zero_padding, up_ref, down_ref, B
            )
            
            if x_sync == -1:
                log_file.write(f"⚠ ERROR: Error en sincronización para trama {frame_idx+1}\n\n")
                all_results.append({
                    'frame_idx': frame_idx,
                    'status': 'SYNC_FAILED',
                    'error': 'Error en sincronización'
                })
                current_pos = actual_start + frame_info['frame_length']
                continue
            
            log_file.write("✓ Sincronización exitosa:\n")
            log_file.write(f"  - x_sync: {x_sync}\n")
            log_file.write(f"  - preamble_bin: {preamble_bin}\n")
            log_file.write(f"  - preamble_bin_zp: {preamble_bin_zp}\n")
            log_file.write(f"  - CFO estimado: {cfo_hz:.2f} Hz\n\n")
            
            # Comparación CFO
            cfo_bins_inj = frame_info['cfo_hz_frame'] * M / B
            cfo_bins_est_cont = (cfo_hz or 0.0) * M / B
            cfo_error_hz = abs(cfo_hz - frame_info['cfo_hz_frame']) if cfo_hz else None
            cfo_error_bins = abs(cfo_bins_est_cont - cfo_bins_inj)
            
            log_file.write("Análisis de CFO:\n")
            log_file.write(f"  - CFO inyectado: {frame_info['cfo_hz_frame']:.2f} Hz ({cfo_bins_inj:.2f} bins)\n")
            log_file.write(f"  - CFO estimado: {cfo_hz:.2f} Hz ({cfo_bins_est_cont:.2f} bins)\n")
            log_file.write(f"  - Error CFO: {cfo_error_hz:.2f} Hz ({cfo_error_bins:.3f} bins)\n")
            
            def wrap_frac(x): return ((x + 0.5) % 1.0) - 0.5
            pre_signed = preamble_bin if preamble_bin <= M//2 else preamble_bin - M
            log_file.write(f"  - [CFO] entero(ref)={pre_signed:+.0f} | "
                          f"frac inyectado={wrap_frac(cfo_bins_inj-pre_signed):+.3f} | "
                          f"frac estimado={wrap_frac(cfo_bins_est_cont-pre_signed):+.3f}\n")
            
            cfo_bins_est_zp = preamble_bin_zp / float(zero_padding)
            log_file.write(f"  - [ZP] preamble_bin_zp={preamble_bin_zp} -> CFO_bins_est_ZP={cfo_bins_est_zp:.3f}\n\n")
            
            #===========================================================================
            #                               DEMODULACIÓN
            #===========================================================================
            symbols_rx, num_avail = rx.demod_data(
                tx_signal[current_pos:], x_sync, num_symbols, M, zero_padding, 
                up_ref, preamble_bin_zp
            )
            
            # Métricas de desempeño
            tx_sym_chunk = frame_info['symbols_tx'][:num_avail]
            rx_sym_chunk = symbols_rx[:num_avail]
            num_symbol_errors = np.sum(tx_sym_chunk != rx_sym_chunk)
            SER = num_symbol_errors / max(1, num_avail)
            
            # Bits
            nbits_proc = num_avail * SF
            tx_bits_chunk = frame_info['bits_tx'][:nbits_proc]
            rx_bits_chunk = rx.decode_symbols_to_bits(rx_sym_chunk, SF)[:nbits_proc]
            num_bit_errors = np.sum(tx_bits_chunk != rx_bits_chunk)
            BER = num_bit_errors / max(1, nbits_proc)
            
            # Almacenar resultados
            frame_result = {
                'frame_idx': frame_idx,
                'status': 'SUCCESS',
                'preamble_start': actual_start,
                'expected_start': expected_start,
                'sync_offset': actual_start - expected_start,
                'cfo_hz_actual': frame_info['cfo_hz_frame'],
                'cfo_hz_estimated': cfo_hz,
                'cfo_error_hz': cfo_error_hz,
                'cfo_error_bins': cfo_error_bins,
                'SER': SER,
                'BER': BER,
                'num_symbol_errors': num_symbol_errors,
                'num_bit_errors': num_bit_errors,
                'symbols_processed': num_avail,
                'bits_processed': nbits_proc
            }
            
            all_results.append(frame_result)
            
            # Mostrar resultados de esta trama
            log_file.write(f"--- RESULTADOS TRAMA {frame_idx+1} ---\n")
            log_file.write(f"Símbolos procesados: {num_avail}/{num_symbols}\n")
            log_file.write(f"SER: {SER:.6f} ({num_symbol_errors} errores)\n")
            log_file.write(f"BER: {BER:.6f} ({num_bit_errors} errores)\n\n")
            
            # Actualizar posición para buscar la siguiente trama
            # Usar la longitud real de la trama actual
            current_pos = actual_start + frame_info['frame_length']
        
        return all_results

    # Procesar todas las tramas
    results = process_multiple_frames(
        tx_signal, frames_info, M, zero_padding, up_ref, down_ref, B, 
        num_symbols, SF, rf_freq, log_file
    )

    #===========================================================================
    #                         RESUMEN FINAL
    #===========================================================================

    log_file.write(f"\n{'='*80}\n")
    log_file.write("RESUMEN FINAL DE TODAS LAS TRAMAS\n")
    log_file.write(f"{'='*80}\n\n")

    successful_frames = [r for r in results if r['status'] == 'SUCCESS']
    failed_frames = [r for r in results if r['status'] != 'SUCCESS']

    log_file.write(f"Tramas exitosas: {len(successful_frames)}/{num_frames}\n")
    log_file.write(f"Tramas fallidas: {len(failed_frames)}/{num_frames}\n")
    log_file.write(f"Tasa de éxito: {100*len(successful_frames)/num_frames:.1f}%\n")

    if successful_frames:
        # Estadísticas agregadas
        avg_ser = np.mean([r['SER'] for r in successful_frames])
        avg_ber = np.mean([r['BER'] for r in successful_frames])
        avg_sync_offset = np.mean([abs(r['sync_offset']) for r in successful_frames])
        avg_cfo_error_hz = np.mean([r['cfo_error_hz'] for r in successful_frames 
                                    if r['cfo_error_hz'] is not None])
        avg_cfo_error_bins = np.mean([r['cfo_error_bins'] for r in successful_frames])
        
        total_symbols = sum([r['symbols_processed'] for r in successful_frames])
        total_symbol_errors = sum([r['num_symbol_errors'] for r in successful_frames])
        total_bits = sum([r['bits_processed'] for r in successful_frames])
        total_bit_errors = sum([r['num_bit_errors'] for r in successful_frames])
        
        log_file.write(f"\n--- PROMEDIOS (TRAMAS EXITOSAS) ---\n")
        log_file.write(f"SER promedio: {avg_ser:.6f}\n")
        log_file.write(f"BER promedio: {avg_ber:.6f}\n")
        log_file.write(f"Error de sincronización promedio: {avg_sync_offset:.2f} muestras\n")
        log_file.write(f"Error CFO promedio: {avg_cfo_error_hz:.2f} Hz ({avg_cfo_error_bins:.3f} bins)\n")
        
        log_file.write(f"\n--- TOTALES AGREGADOS ---\n")
        log_file.write(f"Símbolos totales: {total_symbols}\n")
        log_file.write(f"Errores de símbolo: {total_symbol_errors}\n")
        log_file.write(f"SER agregado: {total_symbol_errors/total_symbols:.6f}\n")
        log_file.write(f"Bits totales: {total_bits}\n")
        log_file.write(f"Errores de bit: {total_bit_errors}\n")
        log_file.write(f"BER agregado: {total_bit_errors/total_bits:.6f}\n")
        
        log_file.write(f"\n--- DETALLE POR TRAMA ---\n")
        log_file.write(f"{'Trama':<8} {'Símbolos':<10} {'SER':<12} {'BER':<12} {'CFO Error (Hz)':<16} {'Sync Error':<12}\n")
        log_file.write(f"{'-'*80}\n")
        for r in successful_frames:
            log_file.write(f"{r['frame_idx']+1:<8} {r['symbols_processed']:<10} {r['SER']:<12.6f} "
                  f"{r['BER']:<12.6f} {r['cfo_error_hz']:<16.2f} {r['sync_offset']:<12} samples\n")

    if failed_frames:
        log_file.write(f"\n--- TRAMAS FALLIDAS ---\n")
        for result in failed_frames:
            log_file.write(f"Trama {result['frame_idx']+1}: {result['status']} - {result.get('error', 'Unknown error')}\n")

    log_file.write(f"\n{'='*80}\n")

    # También imprimir resumen en consola
    print(f"\n{'='*80}")
    print("RESUMEN FINAL DE TODAS LAS TRAMAS")
    print(f"{'='*80}\n")

    print(f"Tramas exitosas: {len(successful_frames)}/{num_frames}")
    print(f"Tramas fallidas: {len(failed_frames)}/{num_frames}")
    print(f"Tasa de éxito: {100*len(successful_frames)/num_frames:.1f}%")

    if successful_frames:
        print(f"\n--- PROMEDIOS (TRAMAS EXITOSAS) ---")
        print(f"SER promedio: {avg_ser:.6f}")
        print(f"BER promedio: {avg_ber:.6f}")
        print(f"Error de sincronización promedio: {avg_sync_offset:.2f} muestras")
        print(f"Error CFO promedio: {avg_cfo_error_hz:.2f} Hz ({avg_cfo_error_bins:.3f} bins)")
        
        print(f"\n--- TOTALES AGREGADOS ---")
        print(f"Símbolos totales: {total_symbols}")
        print(f"Errores de símbolo: {total_symbol_errors}")
        print(f"SER agregado: {total_symbol_errors/total_symbols:.6f}")
        print(f"Bits totales: {total_bits}")
        print(f"Errores de bit: {total_bit_errors}")
        print(f"BER agregado: {total_bit_errors/total_bits:.6f}")
        
        print(f"\n--- DETALLE POR TRAMA ---")
        print(f"{'Trama':<8} {'Símbolos':<10} {'SER':<12} {'BER':<12} {'CFO Error (Hz)':<16} {'Sync Error':<12}")
        print(f"{'-'*80}")
        for r in successful_frames:
            print(f"{r['frame_idx']+1:<8} {r['symbols_processed']:<10} {r['SER']:<12.6f} "
                  f"{r['BER']:<12.6f} {r['cfo_error_hz']:<16.2f} {r['sync_offset']:<12} samples")

    if failed_frames:
        print(f"\n--- TRAMAS FALLIDAS ---")
        for result in failed_frames:
            print(f"Trama {result['frame_idx']+1}: {result['status']} - {result.get('error', 'Unknown error')}")

    print(f"\n{'='*80}\n")
    print(f"Resultados detallados guardados en: {output_file}")

finally:
    # Asegurarse de que el archivo se cierre correctamente
    log_file.close()