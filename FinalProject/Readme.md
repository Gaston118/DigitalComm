# Modulación FSCM (LoRa) en Canales Planos y Selectivos

## 📜 Contexto: Paper de Referencia

Este trabajo está basado en el artículo:

> **"Frequency Shift Chirp Modulation: The LoRa Modulation"**  
> *Lorenzo Vangelista, IEEE Signal Processing Letters, Vol. 24, No. 12, 2017*  

En este artículo se realiza, por primera vez, una descripción teórica rigurosa del esquema de modulación usado por LoRa: **Frequency Shift Chirp Modulation (FSCM)**. Además, se compara el rendimiento en términos de **BER (Bit Error Rate)** entre FSCM y FSK en canales:

- **AWGN (Flat):** canal aditivo con ruido blanco gaussiano.
- **Selectivo en frecuencia:** canal con multitrayectoria (multipath) que atenúa distintas frecuencias de manera desigual.

---

## 📚 Teoría Clave del Paper

### ✅ 1. Modulación FSCM

Para un símbolo s ∈ {0, 1, ..., 2^SF - 1}, la señal transmitida es:

```
c[n] = (1/√(2^SF)) · exp(j·2π·[((s + n) mod 2^SF)/2^SF]·n)
```

donde:
- **SF**: *Spreading Factor* (factor de dispersión, típico entre 7 y 12)
- **n**: índice de muestra en el intervalo de símbolo
- Esta fórmula genera una **chirp ascendente con un corrimiento de frecuencia** proporcional al símbolo transmitido

### ✅ 2. Demodulación óptima

El proceso de demodulación consiste en:

1. **Multiplicación**: Se multiplica la señal recibida por un **chirp descendente** (down-chirp), que es el inverso del chirp base
2. **Transformada de Fourier**: Se calcula la **FFT** del resultado
3. **Detección**: El índice del pico de la FFT indica el símbolo transmitido

### ✅ 3. Modelo de canal selectivo

El canal multipath considerado tiene una respuesta al impulso:

```
h[n] = √(0.8) · δ[n] + √(0.2) · δ[n - 1]
```

Este canal introduce:
- **Interferencia intersimbólica (ISI)**
- **Atenuación selectiva en frecuencia**

Esto afecta negativamente al FSK pero **no tanto al FSCM**, que barre todo el ancho de banda.

---

## 💻 Implementación en Python

### 📐 Parámetros de simulación

| Parámetro                | Valor        | Descripción                    |
|--------------------------|--------------|--------------------------------|
| SF                       | 7            | Spreading Factor               |
| Cardinalidad (M)         | 128 símbolos | 2^SF                          |
| Ancho de banda (B)       | 125 kHz      | Ancho de banda del canal      |
| Muestras por símbolo     | 128          | Resolución temporal           |
| Número de símbolos       | 20,000       | Para estadísticas confiables  |
| Número de bits           | 140,000      | SF × Número de símbolos       |

### ⚙️ Flujo de simulación

1. **Generación de datos**: Creación de bits aleatorios para transmitir

2. **Codificación LoRa**: Los bits se agrupan de a 7 (SF=7) para formar símbolos enteros

3. **Modulación FSCM**: Aplicación de la fórmula del chirp para cada símbolo

4. **Canal de transmisión**:
   - Adición de ruido AWGN en distintos niveles de SNR (Eb/N0)
   - Aplicación del canal multipath (para el caso selectivo)

5. **Demodulación**:
   - Multiplicación por el chirp descendente
   - Cálculo de la FFT
   - Detección del símbolo como el índice del máximo

6. **Decodificación**: Conversión de símbolos de vuelta a bits

7. **Análisis de rendimiento**:
   - Cálculo de tasas de error BER y SER
   - Generación de curvas de rendimiento

---

## 📊 Resultados Esperados

El script debe producir gráficas que muestren:

### Curvas de rendimiento típicas:

- **Eje X**: SNR (Signal-to-Noise Ratio) en dB
- **Eje Y**: Tasa de error (BER/SER) en escala logarítmica
- **Líneas**:
  - BER en canal AWGN (plano)
  - SER en canal AWGN (plano)
  - BER en canal selectivo en frecuencia
  - SER en canal selectivo en frecuencia

### Observaciones clave esperadas:

1. **Canal plano (AWGN)**: BER y SER son similares para FSCM, concordando con los resultados del paper

2. **Canal selectivo en frecuencia**:
   - La BER de FSCM es significativamente menor que la SER
   - Esto confirma la **robustez superior** de FSCM
   - Demuestra la ventaja del barrido de espectro (chirp) para combatir la distorsión selectiva

3. **Comparación general**: Las curvas muestran claramente la **superioridad del FSCM** sobre modulaciones convencionales en canales con desvanecimiento selectivo

---

## 📈 Comparación con el Paper Original

| Aspecto                       | Paper de Vangelista | Implementación Propia |
|-------------------------------|--------------------|-----------------------|
| Modulación                    | FSCM               | FSCM                  |
| Canal plano (AWGN)           | ✔️                 | ✔️                    |
| Canal selectivo en frecuencia | ✔️                 | ✔️                    |
| Métricas mostradas           | Solo BER           | BER y SER             |
| Rango de SNR                 | -12 dB a -1 dB     | -10 dB a +6 dB        |
| Modelo de canal              | Multipath simple   | Multipath simple      |

---

## 🔬 Conceptos Técnicos Clave

### Frequency Shift Chirp Modulation (FSCM)

- **Principio**: Cada símbolo se representa por un chirp (barrido de frecuencia) con un desplazamiento inicial específico
- **Ventaja**: El barrido completo del ancho de banda proporciona diversidad en frecuencia
- **Robustez**: Resistente a la interferencia selectiva en frecuencia

### Spreading Factor (SF)

- **Definición**: Determina la duración del símbolo y la cantidad de información por símbolo
- **Relación**: SF = log₂(M), donde M es el número de símbolos posibles
- **Compromiso**: Mayor SF → mayor robustez pero menor tasa de datos

### Canal Selectivo en Frecuencia

- **Causa**: Multitrayectoria en la propagación de la señal
- **Efecto**: Diferentes frecuencias experimentan atenuaciones distintas
- **Impacto**: Degrada el rendimiento de modulaciones de banda estrecha (como FSK)

---

## 🧪 Extensiones Posibles

### Mejoras en la simulación:
- ✅ Incluir modulación **FSK** para comparación directa
- ✅ Simular diferentes **valores de SF** (8, 9, 10, 11, 12)
- ✅ Implementar **canales más realistas**: Rayleigh, Rice, con efecto Doppler

### Análisis adicionales:
- ✅ Evaluar rendimiento **con codificación de canal** (Hamming, LoRaWAN FEC)
- ✅ Implementar detección **no coherente** o basada en energía
- ✅ Analizar el **espectro de potencia** de las señales FSCM

### Optimizaciones:
- ✅ Estudiar el **sincronismo** y estimación de canal
- ✅ Implementar **técnicas de ecualización**
- ✅ Analizar el **rendimiento en redes multi-usuario**

---

## 📚 Fundamentos Matemáticos

### Señal Chirp Base

La señal chirp fundamental utilizada en LoRa es:

```
c₀[n] = exp(j·2π·(n²/2N)), n = 0, 1, ..., N-1
```

donde N = 2^SF es el número de muestras por símbolo.

### Modulación por Desplazamiento Circular

Para transmitir un símbolo s, se aplica un desplazamiento circular:

```
cₛ[n] = c₀[(n + s) mod N]
```

### Correlación Cruzada

La demodulación se basa en la correlación cruzada entre la señal recibida y el chirp de referencia, aprovechando las propiedades de ortogonalidad de los chirps desplazados.

---

## 🔍 Análisis de Complejidad

### Complejidad computacional:
- **Modulación**: O(N) por símbolo
- **Demodulación**: O(N log N) por símbolo (debido a la FFT)
- **Total**: Dominado por la FFT en la demodulación

### Eficiencia espectral:
- **Tasa de bits**: SF bits por símbolo
- **Ancho de banda**: Fijo (125 kHz típico)
- **Eficiencia**: Decrece con SF mayor, pero aumenta la robustez

---

## 📌 Referencias y Recursos

### Referencia principal:
1. **Vangelista, L. (2017)**. *"Frequency Shift Chirp Modulation: The LoRa Modulation"*. IEEE Signal Processing Letters, Vol. 24, No. 12.

### Referencias complementarias:
2. **LoRa Alliance**. *LoRaWAN Specification v1.0.4*
3. **Semtech Corporation**. *LoRa Modulation Basics*
4. **Reynders, B., et al. (2016)**. *"Chirp spread spectrum as a modulation technique for long range communication"*

### Herramientas de simulación:
- **Python**: NumPy, SciPy, Matplotlib
- **MATLAB**: Communications Toolbox
- **GNU Radio**: Implementación en tiempo real

---

## 💡 Conclusiones

La modulación FSCM utilizada en LoRa demuestra ventajas significativas en canales con desvanecimiento selectivo en frecuencia debido a:

1. **Diversidad en frecuencia**: El barrido completo del ancho de banda
2. **Robustez ante interferencia**: Resistencia a la interferencia de banda estrecha
3. **Simplicidad de implementación**: Demodulación eficiente mediante FFT
4. **Escalabilidad**: Ajuste flexible del compromiso robustez-tasa de datos mediante SF

Estas características hacen de FSCM una elección excelente para aplicaciones IoT de largo alcance y baja potencia.
