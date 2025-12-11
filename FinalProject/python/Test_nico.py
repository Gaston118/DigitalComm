import time

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
NUM_FRAMES = 20           # Cantidad de tramas a probar
SF = 7
M = 2**SF
B = 125e3
T = 1/B
num_symbols = 2000        # Símbolos de datos por trama
num_bits = num_symbols * SF
Ns = 2 * M 

# Frecuencia de Portadora (Para simular SFO físico)
RF_FREQ = 915e6           

# Referencias
up_ref = waveform_former(0, M, B, T)
down_ref = make_down_ref(M, B, T)
zero_padding = 10

# Configuración de Ruido/Silencio
NOISE_FLOOR_AMP = 0.0     # ← CAMBIO: 0.0 para canal ideal sin ruido
MAX_SILENCE_SYMS = 50     # Silencio aleatorio entre 5 y 50 símbolos

# LDRO
ldr_active = (2**SF / B) > 16e-3

# Contadores Globales
total_frames_ok = 0
total_frames_sent = 0
accum_bit_errors = 0
accum_bits_sent = 0
accum_symbol_errors = 0
accum_symbols_sent = 0

# Estadísticas de CFO
cfo_errors = []

print(f"=== STRESS TEST LoRa (Canal Ideal + SFO/CFO) ===")
print(f"Config: {NUM_FRAMES} tramas, SF{SF}, Símbolos/trama={num_symbols}")
print(f"Frecuencia RF: {RF_FREQ/1e6} MHz")
print(f"Ruido Base: {NOISE_FLOOR_AMP} (Canal {'IDEAL' if NOISE_FLOOR_AMP == 0 else 'con ruido'})")
print(f"LDRO: {'ACTIVO' if ldr_active else 'INACTIVO'}")
print("-" * 120)
print(f"{'#':<4} | {'SILENCIO':<10} | {'CFO inj':<10} | {'CFO est':<10} | {'Error CFO':<10} | {'BER':<12} | {'SER':<12} | {'ESTADO'}")
print("-" * 120)

start_time = time.time()

for i in range(NUM_FRAMES):
    # =========================================================================
    # 1. GENERACIÓN DE TRAMA
    # =========================================================================
    bits_tx = np.random.randint(0, 2, size=num_bits)
    symbols_tx = encode_bits_to_symbols(bits_tx, SF)
    tx_clean = lora_modulate(symbols_tx, M, B, T)
    
    # =========================================================================
    # 2. INYECCIÓN DE CFO/SFO
    # =========================================================================
    # Rango de CFO: [-2, +2] bins (fraccionario)
    cfo_bins_inj = np.random.uniform(-2.0, 2.0)
    cfo_hz_inj = cfo_bins_inj * B / M
    
    # Aplicar CFO/SFO (rotación de fase + estiramiento temporal)
    tx_cfo = inject_sfo_cfo(tx_clean, cfo_hz_inj, B, RF_FREQ)
    
    # =========================================================================
    # 3. SILENCIO ALEATORIO (simula inicio no sincronizado)
    # =========================================================================
    silence_syms = np.random.randint(5, MAX_SILENCE_SYMS)
    silence_samples = silence_syms * Ns
    
    if NOISE_FLOOR_AMP > 0:
        noise = (np.random.randn(silence_samples) + 1j*np.random.randn(silence_samples)) / np.sqrt(2)
        silence_sig = NOISE_FLOOR_AMP * noise
    else:
        # Canal ideal: silencio = ceros exactos
        silence_sig = np.zeros(silence_samples, dtype=complex)
    
    # Señal recibida: [Silencio] + [Trama con CFO/SFO]
    rx_signal = np.concatenate([silence_sig, tx_cfo])
    
    # =========================================================================
    # 4. RECEPCIÓN: DETECCIÓN
    # =========================================================================
    x_det = detect(rx_signal, 0, Ns, 8, M, zero_padding, up_ref)
    
    if x_det == -1:
        total_frames_sent += 1
        print(f"{i+1:<4} | {silence_samples:<10} | {cfo_bins_inj:+.3f} bins | {'---':<10} | {'---':<10} | {'---':<12} | {'---':<12} | ❌ NO DETECTADO")
        continue

    # =========================================================================
    # 5. RECEPCIÓN: SINCRONIZACIÓN (con sync_fixed)
    # =========================================================================
    x_sync, pre_bin, pre_bin_zp, cfo_est = sync(
        rx_signal, x_det, Ns, M, zero_padding, up_ref, down_ref, B
    )
    
    if x_sync == -1:
        total_frames_sent += 1
        print(f"{i+1:<4} | {silence_samples:<10} | {cfo_bins_inj:+.3f} bins | {'---':<10} | {'---':<10} | {'---':<12} | {'---':<12} | ❌ FALLO SYNC")
        continue
    
    # =========================================================================
    # 6. RECEPCIÓN: DEMODULACIÓN (con compensación SFO)
    # =========================================================================
    symbols_rx, num_avail = demod_data(
        rx_signal, x_sync, num_symbols, M, zero_padding, 
        up_ref, pre_bin_zp, cfo_est, B, RF_FREQ, ldr=ldr_active
    )
    
    if num_avail == 0:
        total_frames_sent += 1
        print(f"{i+1:<4} | {silence_samples:<10} | {cfo_bins_inj:+.3f} bins | {(cfo_est*M/B):+.3f} bins | {'---':<10} | {'---':<12} | {'---':<12} | ❌ DEMOD VACÍA")
        continue
    
    # =========================================================================
    # 7. CÁLCULO DE MÉTRICAS
    # =========================================================================
    # Recortar a la longitud procesada
    limit = min(len(symbols_tx), num_avail)
    sym_tx_cut = symbols_tx[:limit]
    sym_rx_cut = symbols_rx[:limit]
    
    # Errores de Símbolo
    errs_sym = np.sum(sym_tx_cut != sym_rx_cut)
    ser_frame = errs_sym / limit if limit > 0 else 0.0
    
    # Errores de Bit
    bits_rx = decode_symbols_to_bits(sym_rx_cut, SF)
    bits_tx_cut = bits_tx[:len(bits_rx)]
    errs_bit = np.sum(bits_tx_cut != bits_rx)
    ber_frame = errs_bit / len(bits_tx_cut) if len(bits_tx_cut) > 0 else 0.0
    
    # Error de CFO (en bins)
    cfo_bins_est = (cfo_est * M / B) if cfo_est is not None else 0.0
    cfo_error_bins = abs(cfo_bins_inj - cfo_bins_est)
    
    # =========================================================================
    # 8. ACUMULADORES GLOBALES
    # =========================================================================
    accum_bit_errors += errs_bit
    accum_bits_sent += len(bits_tx_cut)
    accum_symbol_errors += errs_sym
    accum_symbols_sent += limit
    total_frames_sent += 1
    cfo_errors.append(cfo_error_bins)
    
    if errs_sym == 0 and errs_bit == 0:
        total_frames_ok += 1
        status = "✅ PERFECTO"
    elif errs_sym < 10:
        status = f"⚠️ {errs_sym} Sym, {errs_bit} Bits"
    else:
        status = f"❌ {errs_sym} Sym, {errs_bit} Bits"
    
    # =========================================================================
    # 9. IMPRIMIR RESULTADO DE LA TRAMA
    # =========================================================================
    print(f"{i+1:<4} | {silence_samples:<10} | {cfo_bins_inj:+.3f} bins | {cfo_bins_est:+.3f} bins | {cfo_error_bins:.6f} bins | {ber_frame:.9f} | {ser_frame:.9f} | {status}")

# =============================================================================
# REPORTE FINAL
# =============================================================================
elapsed = time.time() - start_time
ber_global = accum_bit_errors / (accum_bits_sent + 1e-12)
ser_global = accum_symbol_errors / (accum_symbols_sent + 1e-12)
cfo_error_mean = np.mean(cfo_errors) if cfo_errors else 0.0
cfo_error_max = np.max(cfo_errors) if cfo_errors else 0.0

print("-" * 120)
print(f"\n{'='*120}")
print(f"REPORTE FINAL - Stress Test Completado")
print(f"{'='*120}")
print(f"Tiempo de ejecución:     {elapsed:.2f}s")
print(f"Tramas procesadas:       {total_frames_sent}/{NUM_FRAMES}")
print(f"Tramas perfectas:        {total_frames_ok}/{total_frames_sent} ({(total_frames_ok/max(1,total_frames_sent))*100:.1f}%)")
print(f"-" * 120)
print(f"BER Global:              {ber_global:.10f}  ({accum_bit_errors}/{accum_bits_sent} bits)")
print(f"SER Global:              {ser_global:.10f}  ({accum_symbol_errors}/{accum_symbols_sent} símbolos)")
print(f"-" * 120)
print(f"Error CFO Promedio:      {cfo_error_mean:.6f} bins")
print(f"Error CFO Máximo:        {cfo_error_max:.6f} bins")
print(f"{'='*120}")

# =============================================================================
# VEREDICTO
# =============================================================================
if NOISE_FLOOR_AMP == 0.0:
    # Canal ideal: debe ser perfecto
    if ber_global == 0.0 and ser_global == 0.0 and cfo_error_mean < 0.01:
        print(f"\n🎉 ¡ÉXITO TOTAL! Canal ideal funcionando perfectamente.")
        print(f"   ✓ BER = 0")
        print(f"   ✓ SER = 0")
        print(f"   ✓ Error CFO < 0.01 bins")
    else:
        print(f"\n⚠️ CANAL IDEAL CON ERRORES RESIDUALES:")
        if ber_global > 0:
            print(f"   ✗ BER = {ber_global:.10f} (esperado: 0)")
        if ser_global > 0:
            print(f"   ✗ SER = {ser_global:.10f} (esperado: 0)")
        if cfo_error_mean >= 0.01:
            print(f"   ✗ Error CFO = {cfo_error_mean:.6f} bins (esperado: < 0.01)")
else:
    # Canal con ruido: evaluar degradación
    print(f"\n📊 CANAL CON RUIDO (Amp={NOISE_FLOOR_AMP}):")
    print(f"   BER = {ber_global:.6f}")
    print(f"   SER = {ser_global:.6f}")
    if ber_global < 0.01:
        print(f"   ✓ Rendimiento excelente (BER < 1%)")
    elif ber_global < 0.05:
        print(f"   ⚠️ Rendimiento aceptable (BER < 5%)")
    else:
        print(f"   ❌ Rendimiento degradado (BER ≥ 5%)")

print(f"{'='*120}\n")