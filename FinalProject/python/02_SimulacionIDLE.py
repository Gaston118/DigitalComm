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
num_symbols = 2000                                  # Número de símbolos por trama
num_bits = num_symbols * SF                         # Número de bits por trama
num_frames = 5                                      # Número de tramas a generar
silence_duration = 1000                             # Duración del silencio entre tramas (muestras)
output_filename = f"02_simulacion_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

#===========================================================================================
#                       GENERACIÓN DE MÚLTIPLES TRAMAS LORA
#===========================================================================================

def generate_lora_frame(symbols_tx, M, B, T, cfo_hz_frame):
    """Genera una trama LoRa completa con CFO"""
    preamble, netid, sfd = tx.preamble_netid_sfd(M, B, T)
    data_waveform = np.concatenate([tx.waveform_former(i, M, B, T) for i in symbols_tx])
    tx_signal_sin_cfo = np.concatenate([preamble, netid, sfd, data_waveform])
    return tx.inject_cfo(tx_signal_sin_cfo, cfo_hz_frame, fs_eff=B)

def generate_multiple_frames(num_frames, num_symbols, SF, M, B, T, silence_duration):
    """Genera múltiples tramas LoRa con silencios entre ellas"""
    all_frames = []
    frame_info = []  # Para almacenar información de cada trama
    
    for frame_idx in range(num_frames):
        # Generar bits y símbolos para esta trama
        bits_tx = np.random.randint(0, 2, size=num_symbols * SF)
        symbols_tx = tx.encode_bits_to_symbols(bits_tx, SF)
        
        # Generar CFO único para esta trama
        cfo_bins_frame = np.random.uniform(-2.0, 2.0)
        cfo_hz_frame = cfo_bins_frame * B / M
        
        # Generar la trama LoRa
        frame_signal = generate_lora_frame(symbols_tx, M, B, T, cfo_hz_frame)
        
        # Calcular posición esperada en la señal combinada
        if frame_idx == 0:
            frame_start = 0
        else:
            # Posición anterior + longitud de trama anterior + silencio
            prev_frame_length = len(all_frames[-1])
            frame_start = sum(len(f) for f in all_frames) + silence_duration * frame_idx
        
        # Almacenar información de la trama
        frame_info.append({
            'frame_idx': frame_idx,
            'bits_tx': bits_tx,
            'symbols_tx': symbols_tx,
            'cfo_hz_frame': cfo_hz_frame,
            'cfo_bins_frame': cfo_bins_frame,
            'frame_length': len(frame_signal),
            'expected_start': frame_start
        })
        
        # Agregar silencio antes de la trama (excepto para la primera)
        if frame_idx > 0:
            silence = np.zeros(silence_duration)
            all_frames.append(silence)
        
        # Agregar la trama
        all_frames.append(frame_signal)
    
    # Convertir a array numpy
    combined_signal = np.concatenate(all_frames)
    
    return combined_signal, frame_info

# Generar todas las tramas
tx_signal, frames_info = generate_multiple_frames(
    num_frames, num_symbols, SF, M, B, T, silence_duration
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

up_ref = tx.waveform_former(0, M, B, T)
down_ref = rx.make_down_ref(M, B, T)
zero_padding = 10

#===========================================================================
#                       PROCESAMIENTO Y GENERACIÓN DE REPORTE
#===========================================================================

def write_to_file(filename, content, mode='a'):
    """Escribe contenido en el archivo de resultados"""
    with open(filename, mode, encoding='utf-8') as f:
        f.write(content + '\n')

def process_and_generate_report(tx_signal, frames_info, M, zero_padding, up_ref, down_ref, B, num_symbols, SF, output_filename):
    """Procesa las tramas y genera un reporte completo en archivo TXT"""
    
    # Crear archivo y escribir encabezado
    header = f"""
{'='*80}
RESULTADOS SIMULACIÓN LoRa - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
Configuración:
- SF: {SF}, M: {M}
- Ancho de banda: {B/1000:.0f} kHz
- Símbolos por trama: {num_symbols}
- Bits por trama: {num_symbols * SF}
- Número de tramas: {len(frames_info)}
- Duración de silencio: {silence_duration} muestras
- Zero padding: {zero_padding}

{'='*80}
INFORMACIÓN DE TRAMAS GENERADAS
{'='*80}
"""
    write_to_file(output_filename, header, 'w')
    
    # Información de las tramas generadas
    for frame_info in frames_info:
        frame_line = (f"Trama {frame_info['frame_idx']:2d}: "
                     f"CFO = {frame_info['cfo_hz_frame']:8.2f} Hz "
                     f"({frame_info['cfo_bins_frame']:6.3f} bins), "
                     f"Longitud = {frame_info['frame_length']:6d} muestras, "
                     f"Posición esperada = {frame_info['expected_start']:6d}")
        write_to_file(output_filename, frame_line)
    
    write_to_file(output_filename, f"\nLongitud total de la señal: {len(tx_signal)} muestras")
    
    # Procesamiento de tramas
    all_results = []
    search_start = 0
    
    write_to_file(output_filename, f"\n{'='*80}")
    write_to_file(output_filename, "PROCESAMIENTO DE TRAMAS")
    write_to_file(output_filename, '='*80)
    
    for frame_idx, frame_info in enumerate(frames_info):
        expected_start = frame_info['expected_start']
        frame_length = frame_info['frame_length']
        
        write_to_file(output_filename, f"\nTRAMA {frame_idx}:")
        write_to_file(output_filename, "-" * 40)
        
        # Detección
        x = rx.detect(tx_signal[search_start:], 0, M, 8, M, zero_padding, up_ref, mag_threshold=None)
        
        if x != -1:
            actual_start = search_start + x - (8 - 1) * M
            preamble_start = x - (8 - 1) * M
            
            netid_len = 2 * M
            sfd_len = 2 * M + (M // 4)
            data_start_nom = preamble_start + 8 * M + netid_len + sfd_len
            
            write_to_file(output_filename, f"  Preámbulo detectado en: {search_start + x}")
            write_to_file(output_filename, f"  Inicio real: {actual_start}")
            write_to_file(output_filename, f"  Esperado: {expected_start}")
            write_to_file(output_filename, f"  Diferencia: {actual_start - expected_start:+6d} muestras")
        else:
            write_to_file(output_filename, "  ❌ DETECCIÓN FALLIDA")
            all_results.append({
                'frame_idx': frame_idx,
                'status': 'DETECTION_FAILED',
                'error': 'No se detectó preámbulo'
            })
            search_start += 10000
            continue
        
        # Sincronización
        x_sync, preamble_bin, preamble_bin_zp, cfo_hz = rx.sync(
            tx_signal[search_start:], x, M, M, zero_padding, up_ref, down_ref, B
        )
        
        if x_sync == -1:
            write_to_file(output_filename, "  ❌ SINCRONIZACIÓN FALLIDA")
            all_results.append({
                'frame_idx': frame_idx,
                'status': 'SYNC_FAILED',
                'error': 'Error en sincronización'
            })
            search_start = actual_start + 1000
            continue
        
        # Comparación CFO
        cfo_bins_inj = frame_info['cfo_hz_frame'] * M / B
        cfo_bins_est_cont = (cfo_hz or 0.0) * M / B
        
        write_to_file(output_filename, f"  ✅ Sincronización exitosa")
        write_to_file(output_filename, f"  CFO inyectado:  {cfo_bins_inj:7.3f} bins")
        write_to_file(output_filename, f"  CFO estimado:   {cfo_bins_est_cont:7.3f} bins")
        write_to_file(output_filename, f"  Error CFO:      {abs(cfo_bins_est_cont - cfo_bins_inj):7.3f} bins")
        
        # Demodulación
        symbols_rx, num_avail = rx.demod_data(
            tx_signal[search_start:], x_sync, num_symbols, M, zero_padding, 
            up_ref, preamble_bin_zp
        )
        
        # Métricas
        tx_sym_chunk = frame_info['symbols_tx'][:num_avail]
        rx_sym_chunk = symbols_rx[:num_avail]
        num_symbol_errors = np.sum(tx_sym_chunk != rx_sym_chunk)
        SER = num_symbol_errors / max(1, num_avail)
        
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
            'cfo_error_bins': abs(cfo_bins_est_cont - cfo_bins_inj),
            'SER': SER,
            'BER': BER,
            'num_symbol_errors': num_symbol_errors,
            'num_bit_errors': num_bit_errors,
            'symbols_processed': num_avail
        }
        
        all_results.append(frame_result)
        
        # Escribir resultados de esta trama
        write_to_file(output_filename, f"  Símbolos procesados: {num_avail}/{num_symbols}")
        write_to_file(output_filename, f"  Errores de símbolo: {num_symbol_errors}")
        write_to_file(output_filename, f"  Errores de bit: {num_bit_errors}")
        write_to_file(output_filename, f"  SER: {SER:.6f}")
        write_to_file(output_filename, f"  BER: {BER:.6f}")
        
        # Actualizar posición de búsqueda
        search_start = actual_start + frame_length + silence_duration
    
    return all_results

# Procesar y generar reporte
print(f"Procesando {num_frames} tramas LoRa...")
print(f"Los resultados se guardarán en: {output_filename}")

results = process_and_generate_report(
    tx_signal, frames_info, M, zero_padding, up_ref, down_ref, B, 
    num_symbols, SF, output_filename
)

#===========================================================================
#                         RESUMEN FINAL EN ARCHIVO
#===========================================================================

def generate_final_summary(results, output_filename):
    """Genera el resumen final en el archivo"""
    
    successful_frames = [r for r in results if r['status'] == 'SUCCESS']
    failed_frames = [r for r in results if r['status'] != 'SUCCESS']
    
    summary = f"""
{'='*80}
RESUMEN FINAL
{'='*80}

ESTADÍSTICAS GENERALES:
• Tramas exitosas: {len(successful_frames)}/{len(results)}
• Tramas fallidas:  {len(failed_frames)}/{len(results)}
• Tasa de éxito:    {len(successful_frames)/len(results)*100:.1f}%

"""
    write_to_file(output_filename, summary)
    
    if successful_frames:
        avg_ser = np.mean([r['SER'] for r in successful_frames])
        avg_ber = np.mean([r['BER'] for r in successful_frames])
        avg_sync_offset = np.mean([abs(r['sync_offset']) for r in successful_frames])
        avg_cfo_error = np.mean([r['cfo_error_bins'] for r in successful_frames])
        max_sync_offset = max([abs(r['sync_offset']) for r in successful_frames])
        max_cfo_error = max([r['cfo_error_bins'] for r in successful_frames])
        
        metrics = f"""MÉTRICAS DE TRAMAS EXITOSAS:
• SER promedio:           {avg_ser:.6f}
• BER promedio:           {avg_ber:.6f}
• Error sincronización:   {avg_sync_offset:.2f} muestras (max: {max_sync_offset})
• Error CFO:              {avg_cfo_error:.3f} bins (max: {max_cfo_error:.3f})

"""
        write_to_file(output_filename, metrics)
    
    # Tabla detallada por trama
    write_to_file(output_filename, "DETALLE POR TRAMA:")
    write_to_file(output_filename, "┌───────┬──────────┬────────────┬──────────┬──────────┬──────────────┬────────────┐")
    write_to_file(output_filename, "│ Trama │ Estado   │ SER        │ BER      │ SyncError│ CFO Error    │ Símbolos   │")
    write_to_file(output_filename, "│       │          │            │          │ (muestras)│ (bins)      │ Procesados │")
    write_to_file(output_filename, "├───────┼──────────┼────────────┼──────────┼──────────┼──────────────┼────────────┤")
    
    for result in results:
        if result['status'] == 'SUCCESS':
            status = "✅ EXITO "
            ser_str = f"{result['SER']:.4f}" if result['SER'] < 0.01 else f"{result['SER']:.4f}⚠"
            ber_str = f"{result['BER']:.4f}" if result['BER'] < 0.01 else f"{result['BER']:.4f}⚠"
            sync_str = f"{result['sync_offset']:>+6d}"
            cfo_str = f"{result['cfo_error_bins']:6.3f}"
            symbols_str = f"{result['symbols_processed']:4d}/{num_symbols}"
        else:
            status = "❌ FALLIDA"
            ser_str = "    -    "
            ber_str = "    -    "
            sync_str = "    -    "
            cfo_str = "    -    "
            symbols_str = "    -    "
        
        row = (f"│ {result['frame_idx']:5d} │ {status} │ {ser_str:>10} │ {ber_str:>8} │ {sync_str:>8} │ {cfo_str:>12} │ {symbols_str:>10} │")
        write_to_file(output_filename, row)
    
    write_to_file(output_filename, "└───────┴──────────┴────────────┴──────────┴──────────┴──────────────┴────────────┘")
    
    # Análisis de calidad
    write_to_file(output_filename, f"\nANÁLISIS DE CALIDAD:")
    
    if successful_frames:
        excellent_ser = len([r for r in successful_frames if r['SER'] < 0.001])
        good_ser = len([r for r in successful_frames if 0.001 <= r['SER'] < 0.01])
        poor_ser = len([r for r in successful_frames if r['SER'] >= 0.01])
        
        excellent_ber = len([r for r in successful_frames if r['BER'] < 0.001])
        good_ber = len([r for r in successful_frames if 0.001 <= r['BER'] < 0.01])
        poor_ber = len([r for r in successful_frames if r['BER'] >= 0.01])
        
        quality_analysis = f"""• Calidad SER:
  - Excelente (SER < 0.001): {excellent_ser} tramas
  - Buena (SER < 0.01):     {good_ser} tramas  
  - Pobre (SER ≥ 0.01):     {poor_ser} tramas

• Calidad BER:
  - Excelente (BER < 0.001): {excellent_ber} tramas
  - Buena (BER < 0.01):     {good_ber} tramas
  - Pobre (BER ≥ 0.01):     {poor_ber} tramas
"""
        write_to_file(output_filename, quality_analysis)
    
    # Conclusión
    success_rate = len(successful_frames) / len(results) * 100
    if success_rate == 100:
        conclusion = "✅ EXCELENTE: Todas las tramas procesadas exitosamente"
    elif success_rate >= 80:
        conclusion = "✅ BUENO: Alta tasa de éxito en el procesamiento"
    elif success_rate >= 60:
        conclusion = "⚠️  ACEPTABLE: Tasa de éxito moderada"
    else:
        conclusion = "❌ POBRE: Baja tasa de éxito, revisar algoritmo"
    
    write_to_file(output_filename, f"\nCONCLUSIÓN: {conclusion}")
    write_to_file(output_filename, '='*80)

# Generar resumen final
generate_final_summary(results, output_filename)

print(f"✅ Procesamiento completado!")
print(f"📄 Reporte guardado en: {output_filename}")

# Mostrar solo un resumen muy breve en consola
successful_count = len([r for r in results if r['status'] == 'SUCCESS'])
print(f"\nResumen ejecución:")
print(f"• Tramas procesadas: {len(results)}")
print(f"• Exitosa: {successful_count}")
print(f"• Fallidas: {len(results) - successful_count}")
print(f"• Archivo de resultados: {output_filename}")