# Modulación FSCM (LoRa) en Canales Planos y Selectivos

## 📜 Contexto: Paper de Referencia

Este trabajo está basado en el artículo:

> **"Frequency Shift Chirp Modulation: The LoRa Modulation"**  
> *Lorenzo Vangelista*  

En este artículo se realiza, por primera vez, una descripción teórica rigurosa del esquema de modulación usado por LoRa: **Frequency Shift Chirp Modulation (FSCM)**. Además, se compara el rendimiento en términos de **BER (Bit Error Rate)** entre FSCM y FSK en canales:

- **AWGN (Flat):** canal aditivo con ruido blanco gaussiano.
- **Selectivo en frecuencia:** canal con multitrayectoria (multipath) que atenúa distintas frecuencias de manera desigual.

---

## 🔍 ¿Qué es LoRa?

La modulación utilizada por **LoRa**, una tecnología clave en redes **LPWAN** (Low Power Wide Area Networks), ha sido históricamente poco documentada en términos teóricos. 
Los objetivos principales del paper son:
- Proporcionar por primera vez una **descripción matemática rigurosa** de la modulación LoRa
- Definir formalmente la **Frequency Shift Chirp Modulation (FSCM)**
- Proponer un **demodulador óptimo** de baja complejidad usando FFT
- Comparar el rendimiento frente a FSK en canales planos y selectivos

---

## 📚 Teoría Clave del Paper

### ✅ 1. Spreading Factor (SF)

El **Spreading Factor** es un parámetro fundamental que determina:
- La **cantidad de bits por símbolo**
- Denotado por SF ∈ {7, 8, 9, 10, 11, 12}
- Cada símbolo representa M = 2^SF posibles valores
- Por lo tanto: log₂(M) = SF bits por símbolo

**Ejemplo**: Para SF = 7 → M = 128 símbolos → 7 bits por símbolo

### ✅ 2. Modulación FSCM: Chirps con Corrimiento

Para transmitir un símbolo s ∈ {0, 1, ..., M-1}, se genera una señal tipo **chirp**, cuya frecuencia aumenta linealmente con el tiempo, pero con un **corrimiento inicial de frecuencia** dependiente de s.

#### 🔧 Fórmula de la señal transmitida:

• Parámetros de la Señal:
- **Factor de Dispersión (SF - Spreading Factor)**: Es un parámetro entero clave en LoRa, que generalmente toma valores en {7, 8, 9, 10, 11, 12}
- **Tiempo de Símbolo (Ts):** Un símbolo s(nTs) se envía cada Ts = 2^SF * T
- **Ancho de Banda (B):** Es el ancho de banda del canal utilizado para la transmisión.
- **Tiempo de Muestra (T):** Se envía una muestra cada T = 1/B

```
s(nTs) = Σ(desde h=0 hasta SF-1) w(nTs)h * 2^h
```

• Forma de Onda Transmitida:

```
c(nTs + kT) = (1/√2^SF) * ej2π [ (s(nTs) + k) mod 2^SF ] * k / (2^SF) para k = 0, ..., 2^SF - 1
```

Donde:
- **n** ∈ {0, 1, ..., M-1} es el índice temporal discreto
- **T** = 1/B es el período de muestreo
- **M** = 2^SF: cantidad total de símbolos posibles
- **s**: símbolo a transmitir

**Cada forma de onda difiere de una forma de onda base (que tiene una frecuencia inicial de 0) por un desplazamiento de frecuencia inicial s(nTs), de ahí el nombre FSCM.**

**Dominio de Análisis: Todo el análisis de la modulación FSCM en el paper se realiza en el dominio discreto Z(T) = {..., -T, 0, T, ...}**

#### 📌 Interpretación:
- Es una **señal chirp discreta** con corrimiento de frecuencia
- La información está codificada como un **desplazamiento de frecuencia inicial**
- Se diferencia de los chirps clásicos al ser modulada en frecuencia por la posición s
- El chirp "barre" todo el ancho de banda disponible

### ✅ 3. Ortogonalidad de las Señales Chirp

El paper demuestra matemáticamente que las M señales chirp generadas para cada símbolo son **ortogonales entre sí** en el dominio discreto:

```
⟨ c(nTs + kT)|s(nTs)=i , c(nTs + kT)|s(nTs)=q ⟩ = 0 para i ≠ q i,q ∈ {0,..., 2^SF−1}
```

**Importancia**: Esta ortogonalidad es esencial para asegurar una detección sin interferencia entre símbolos y permite la separación perfecta en condiciones ideales.

### ✅ 4. Demodulación Óptima de Señales FSCM en Canales de Ruido Gaussiano Blanco Aditivo (AWGN)

◦ La señal recibida r(nTs + kT) es la señal transmitida c(nTs + kT) más un ruido gaussiano blanco w(nTs + kT).
◦ El demodulador óptimo consiste en proyectar la señal recibida r(nTs + kT) sobre cada una de las posibles señales c(nTs + kT)|s(nTs)=q, para q = 0, ..., 2^SF - 1.
◦ Luego, se elige la señal c(nTs + kT)|s(nTs)=l para la cual el módulo (cuadrado) de la proyección es máximo. Esta l es la mejor estimación del símbolo transmitido s(nTs)

El proceso óptimo de demodulación, conocido como **"dechirping"**, consiste en:

#### Paso 1: Multiplicar la señal recibida r(nTs + kT) muestra por muestra por una "down-chirp"
```
e^(-j2π k^2 / (2^SF))
```
```
d(nTs + kT) = r(nTs + kT) * e^(-j2π k^2 / (2^SF))
```

#### Paso 2: Transformada de Fourier
- Se calcula la **FFT** de d(nTs + kT)
- El resultado es: **FFT{d(nTs + kT)}**

#### Paso 3: Detección del símbolo
- El índice del **máximo de la FFT** corresponde al símbolo transmitido ŝ
- **ŝ = arg max|FFT{d[n]}|**

#### 🔧 Ventajas del demodulador:
- **Baja complejidad**: O(M log M) usando FFT
- **Óptimo**: Maximiza la relación señal-ruido
- **Robusto**: Efectivo en canales adversos

### ✅ 5. Modelos de Canal Considerados

#### a) **Canal Plano (AWGN)**:
```
r[n] = c[n] + w[n]
```
- Canal aditivo con ruido blanco gaussiano (modelo ideal)
- No afecta frecuencias de forma selectiva
- **w[n]**: ruido gaussiano complejo

#### b) **Canal Selectivo en Frecuencia**:
```
h[n] = √(0.8) · δ[n] + √(0.2) · δ[n-1]
```
- Representa un **canal multipath** con dos trayectorias
- Trayectoria directa: 80% de la energía
- Trayectoria reflejada: 20% de la energía, con retraso de 1 muestra
- Genera **desvanecimiento selectivo** que degrada FSK pero no tanto FSCM

***Canal de Ruido Gaussiano Blanco Aditivo (AWGN) Selectivo en Frecuencia con Trayectos Múltiples:*** Las simulaciones del estudio consideran tanto un canal AWGN plano (sin distorsión de frecuencia) como un canal AWGN selectivo en frecuencia con trayectos múltiples. La expresión h(nT) es la respuesta al impulso de este último, que es un modelo más realista para entornos donde la señal puede llegar al receptor a través de diferentes caminos.

◦ La función $\delta(nT)$ representa un impulso ideal en el tiempo $nT=0$. En este contexto, simboliza el primer trayecto (o trayecto directo) de la señal, que llega al receptor sin retardo adicional significativo.

◦ La función $\delta(nT-T)$ representa un impulso ideal retardado por un tiempo $T$. Esto significa que hay un segundo trayecto (o multitrayecto) de la señal que llega al receptor con un retardo de $T$ segundos respecto al primer trayecto. Este $T$ es el tiempo de muestreo, es decir, $T = 1/B$, donde $B$ es el ancho de banda del canal.

◦ El término $\sqrt{0.8}$ es el factor de amplitud o ganancia del primer trayecto (el que llega sin retardo).

◦ El término $\sqrt{0.2}$ es el factor de amplitud o ganancia del segundo trayecto (el que llega con retardo $T$).

◦ Se especifica que es un canal de "energía unitaria", lo cual se verifica al sumar los cuadrados de las amplitudes: $(\sqrt{0.8})^2 + (\sqrt{0.2})^2 = 0.8 + 0.2 = 1$. Esto implica que, en total, la energía de la señal a través del canal se conserva.

◦ **Implicaciones:** Un canal con múltiples trayectos y retardos diferentes como este causa que la respuesta en frecuencia del canal no sea plana. Es decir, algunas frecuencias se atenuarán más que otras, mientras que otras podrían incluso amplificarse, lo que se conoce como desvanecimiento selectivo en frecuencia.

---

### 📐 Parámetros de simulación

| Parámetro                | Valor        | Descripción                    |
|--------------------------|--------------|--------------------------------|
| SF                       | 7            | Spreading Factor               |
| Cardinalidad (M)         | 128 símbolos | 2^SF                          |
| Ancho de banda (B)       | 125 kHz      | Ancho de banda del canal      |
| Muestras por símbolo     | 128          | Resolución temporal           |
| Número de símbolos       | 20,000       | Para estadísticas confiables  |
| Número de bits           | 140,000      | SF × Número de símbolos       |

### ⚙️ Flujo de simulación detallado

## Configuración Inicial y Parámetros

```python
SF = 7
M = 2**SF
B = 125e3         # Ancho de banda
T = 1/B           # Periodo de muestra
```

**Comparación con el paper:**
- `SF = 7`: Corresponde al **Spreading Factor** del paper, que toma valores típicos {7,8,9,10,11,12}
- `M = 2^SF = 128`: Es el número de símbolos posibles, coincide con la cardinalidad mencionada en el paper
- `B = 125 kHz`: Ancho de banda del canal (parámetro del sistema)
- `T = 1/B`: **Período de muestreo T**, exactamente como se define en el paper: "transmitir una muestra cada T = 1/B"

## Generación de Datos y Codificación

```python
num_symbols = 20000
num_bits = num_symbols * SF
bits_tx = np.random.randint(0, 2, size=num_bits)
```

**Del paper:** Cada símbolo transporta **SF bits**, por lo que necesitamos `SF × num_symbols` bits totales.

```python
encoder = np.array([
    sum(bits_tx[i*SF + j] << (SF-1-j) for j in range(SF))
    for i in range(num_symbols)
])
```

**Comparación con Ecuación (1) del paper:**
```
s(nTs) = Σ(h=0 to SF-1) w(nTs)h · 2^h
```

El código implementa exactamente esta ecuación:
- `bits_tx[i*SF + j]` = `w(nTs)h` (los bits del símbolo)
- `<< (SF-1-j)` = `2^h` (desplazamiento binario equivale a multiplicar por potencia de 2)
- El resultado es `s(nTs)` que toma valores en {0, 1, 2, ..., 2^SF - 1}

## Formación de la Forma de Onda (Waveform Former)

```python
def waveform_former(symbol, M, B, T):
    k = np.arange(M)
    phase = ((symbol + k) % M) * (k * T * B) / M
    chirp_waveform = np.exp(1j * 2 * np.pi * phase) / np.sqrt(M)
    return chirp_waveform
```

**Comparación con Ecuaciones (2)-(3) del paper:**
```
c(nTs + kT) = (1/√2^SF) · e^(j2π[(s(nTs)+k) mod 2^SF]k·T·B/2^SF)
```

Línea por línea:
- `k = np.arange(M)`: Vector k = 0, 1, ..., 2^SF - 1 (índices temporales del símbolo)
- `(symbol + k) % M`: Implementa `(s(nTs) + k) mod 2^SF`
- `(k * T * B) / M`: Implementa `k·T·B/2^SF` (ya que M = 2^SF)
- `np.exp(1j * 2 * np.pi * phase)`: La exponencial compleja e^(j2π·phase)
- `/ np.sqrt(M)`: Factor de normalización `1/√2^SF`

## Transmisión

```python
tx_signal = np.concatenate([waveform_former(i, M, B, T) for i in encoder])
```

Genera la señal completa concatenando las formas de onda de cada símbolo.

## Demodulador Óptimo

```python
def nTuple_former(received_block, M, B, T):
    k = np.arange(M)
    down_chirp = np.exp(-1j * 2 * np.pi * (k * T * B) * k / M)
    reference_chirp = received_block * down_chirp
    spectrum = np.fft.fft(reference_chirp)
    return np.argmax(np.abs(spectrum))
```

**Comparación con Sección III del paper - Implementación Eficiente:**

### Paso 1: Down-chirp
```python
down_chirp = np.exp(-1j * 2 * np.pi * (k * T * B) * k / M)
```

**Del paper - Ecuación después de (16):**
```
d(nTs + kT) = r(nTs + kT) · e^(-j2π k²/2^SF)
```

Comparando:
- `(k * T * B) * k / M` = `k² · T · B / 2^SF`
- Como `T · B = 1` y `M = 2^SF`, tenemos: `k²/2^SF` ✓

### Paso 2: Multiplicación
```python
reference_chirp = received_block * down_chirp
```

Implementa la multiplicación elemento por elemento descrita en el paper.

### Paso 3: FFT y Detección
```python
spectrum = np.fft.fft(reference_chirp)
return np.argmax(np.abs(spectrum))
```

**Del paper:** "tomar la Transformada Discreta de Fourier... y seleccionar la salida de índice p"

El `argmax` encuentra el índice `p` con mayor magnitud, que corresponde al símbolo estimado.

## Simulación Sin Ruido (Validación)

```python
symbols_rx = []
for idx in range(len(encoder)):
    block = tx_signal[idx*M : (idx+1)*M]
    symbol_hat = nTuple_former(block, M, B, T)
    symbols_rx.append(symbol_hat)
```

Demodula símbolo por símbolo y verifica que en condiciones ideales `BER = SER = 0`.

## Decodificación de Bits

```python
decoder = np.array([
    [(symbols_rx[i] >> (SF-1-j)) & 1 for j in range(SF)]
    for i in range(len(symbols_rx))
]).flatten()
```

Proceso inverso al encoder: convierte cada símbolo decimal de vuelta a sus SF bits constituyentes.

## Simulación con AWGN

```python
snr_dB_range = np.arange(10, 30, 2)   # Es/N0 (dB)
EbN0_dB_range = snr_dB_range - 10*np.log10(M)   # Eb/N0 (dB)
```

**Relación Es/N0 vs Eb/N0:**
- `Es`: Energía por símbolo
- `Eb`: Energía por bit = Es/SF
- `Eb/N0 = Es/N0 / SF`
- En dB: `Eb/N0 (dB) = Es/N0 (dB) - 10·log10(SF)`

Definición de energías según la teoría de comunicaciones digitales.

```python
EbN0 = 10**(EbN0_dB / 10)       # Conversión dB a lineal
EsN0 = EbN0 * SF                # Es/N0 lineal
N0    = Es / EsN0               # Densidad de ruido
sigma = np.sqrt(N0/2)           # Desviación típica por dimensión
```

**N0 - Densidad Espectral de Potencia del Ruido**
- N0 es la densidad espectral de potencia del ruido blanco medida en Watts/Hz.
- Representa cuánta potencia de ruido hay por cada Hz de ancho de banda
- En ruido AWGN (Additive White Gaussian Noise), N0 es constante en todas las frecuencias
- La potencia total del ruido en un ancho de banda B es: Potencia_ruido = N0 × B

**Eb/N0 - Relación Energía por Bit a Densidad de Ruido**
- Eb es la energía por bit (Joules por bit), y Eb/N0 es una medida de calidad de señal fundamental
- Es la métrica estándar para comparar sistemas de comunicación digital
- Es independiente de la tasa de bits (a diferencia de SNR)

**Es/N0 - Relación Energía por Símbolo a Densidad de Ruido**
- Es es la energía por símbolo, y Es/N0 mide la calidad por símbolo.

**SNR - Signal-to-Noise Ratio (Relación Señal a Ruido)**
- SNR es la relación entre la potencia de la señal y la potencia del ruido.

```
SNR = (Potencia_señal) / (N0 × B)
Eb/N0 = (Energía_bit) / N0
```

![image](https://github.com/user-attachments/assets/9d80c4c8-e7c0-4705-8ce9-d28632b6a7bd)

**Densidad Espectral de Potencia (PSD)**
La densidad espectral de potencia describe cómo se distribuye la potencia de una señal (o ruido) a través de las frecuencias.

**Ruido AWGN complejo:**
- Cada dimensión (real e imaginaria) tiene varianza `σ² = N0/2`
- Potencia total del ruido = `N0`

## Canal Selectivo en Frecuencia

```python
h_freqsel = np.array([np.sqrt(0.8), np.sqrt(0.2)])  # √0.8 δ[n] + √0.2 δ[n-1]
```

**Del paper - Sección IV:**
> "canal selectivo en frecuencia AWGN multipaso con energía unitaria con respuesta al impulso h(nT) = √0.8δ(nT) + √0.2δ(nT - T)"

```python
tx_faded = np.convolve(tx_signal, h_freqsel, mode='full')[:len(tx_signal)]
```

### Aplica la respuesta al impulso del canal mediante convolución.

◦ Un canal de comunicación se modela como un sistema que altera la señal que lo atraviesa. Cuando se asume que el canal es lineal e invariante en el tiempo **(SLIT)**, su efecto sobre cualquier señal de entrada puede caracterizarse completamente por su respuesta al impulso, $h(nT)$

- La respuesta al impulso $h(nT)$ describe cómo el canal responde a una señal de entrada idealmente corta (un impulso).

- Para cualquier SLIT, la señal de salida es la convolución de la señal de entrada con la respuesta al impulso del sistema. Es decir, si tx_signal es la señal transmitida (entrada al canal) y h_freqsel es la respuesta al impulso del canal, entonces la señal recibida (salida del canal, tx_faded en este caso) se obtiene mediante su convolución.

- Matemáticamente, la convolución simula cómo cada punto de la señal de entrada interactúa con la respuesta del canal para producir la señal de salida.

- El resultado de esta convolución es que la señal original tx_signal se "mezcla" con versiones retardadas y atenuadas de sí misma, lo que simula el efecto de multitrayecto en un canal real. Esta superposición de versiones retardadas de la señal es lo que causa la selectividad en frecuencia del canal, es decir, que diferentes frecuencias se atenúen o amplifiquen de manera distinta.

- La parte mode='full' en np.convolve asegura que se calcule la convolución completa. La longitud de una convolución completa es len(tx_signal) + len(h_freqsel) - 1. La adición [:len(tx_signal)] después de la convolución (mode='full') indica que se está truncando el resultado de la convolución a la longitud de la señal de transmisión original. 

## Conclusión

1. ✅ **Modulación FSCM** según ecuaciones (2)-(3)
2. ✅ **Demodulador óptimo** con down-chirp y FFT
3. ✅ **Canal AWGN** con ruido gaussiano apropiado
4. ✅ **Canal selectivo** con la misma respuesta al impulso
5. ✅ **Métricas de rendimiento** BER y SER
6. ✅ **Comparación experimental** replicando la Figura 1

---

### Observaciones clave del paper:

#### 1. **Canal AWGN (Plano)**:
- FSCM y FSK tienen **rendimiento similar**
- BER y SER siguen curvas típicas de modulación M-aria
- Concordancia con la teoría clásica de comunicaciones

#### 2. **Canal Selectivo en Frecuencia**:
- **FSCM supera claramente a FSK**
- La BER de FSCM es significativamente menor que FSK
- Ventaja se debe al **barrido completo del espectro**

#### 3. **Robustez de FSCM**:
- El barrido de frecuencias **promedia los efectos del canal**
- Resistente a desvanecimientos selectivos
- Mantiene rendimiento en condiciones adversas

---

## 📈 Comparación Detallada: FSCM vs FSK

| Característica              | FSCM (LoRa)                            | FSK Tradicional                       |
|-----------------------------|--------------------------------------- |---------------------------------------|
| **Base matemática**         | Chirp con desplazamiento               |  Portadoras sinusoidales               |
| **Ortogonalidad**           | ✔️ Sí (demostrada en el paper)        | ✔️ Sí (por frecuencia)                |
| **Robustez en multipath**   | 🟢 Alta (barre todas las frecuencias) |🔴 Baja (puede caer en frecuencias atenuadas) |
| **Complejidad demodulación**| Baja (uso de FFT)                      | Media (banco de correladores)         |
| **Ancho de banda**          | Totalmente utilizado                   | Utiliza una porción por símbolo       |
| **Diversidad en frecuencia**| ✔️ Inherente                          | ❌ No disponible                      |
| **Sincronización**          | Menos crítica                          | Más crítica                           |

---

## AWGN = Additive White Gaussian Noise

- Additive (Aditivo): Se suma a la señal → señal_recibida = señal_transmitida + ruido
- White (Blanco): Densidad espectral plana → N0 constante en todas las frecuencias. Implica que el ruido tiene una densidad espectral de potencia constante en todas las frecuencias dentro del ancho de banda de interés. Es decir, contiene todas las frecuencias con la misma intensidad, de forma análoga a la luz blanca que contiene todos los colores. El paper especifica que es "zero mean white gaussian noise, with $\sigma^2_w (nTs +kT )$ = $\sigma^2_w$ independent of (nTs + kT )"
- Gaussian (Gaussiano): Amplitudes siguen distribución normal
- Noise (Ruido): Señal no deseada que degrada la comunicación
 
**Propiedades del Ruido n(t):**

1) Media cero: E[n(t)] = 0
2) Gaussiano: n(t) ~ N(0, σ²)
3) Blanco: Densidad espectral = N0 (constante)
4) Estacionario: Propiedades no cambian en el tiempo

**Propiedades del Ruido Complejo:**

- n_I(t) y n_Q(t) son independientes
- Ambos son gaussianos con media cero
- Varianza por componente: σ² = N0/2
- Potencia total: E[|n(t)|²] = E[n_I²] + E[n_Q²] = N0/2 + N0/2 = N0

```
Potencia_total = E[|n|²] = E[n_I² + n_Q²] = E[n_I²] + E[n_Q²]

Como n_I y n_Q son independientes con varianza σ²:
E[n_I²] = σ² 
E[n_Q²] = σ²

Por tanto:
N0 = 2σ²  →  σ² = N0/2  →  σ = √(N0/2)

Densidad Espectral de Potencia
     ^
     |     Señal LoRa
     |      /\  /\
     | S(f)/  \/  \
     |    /        \
 N0  |████████████████████████  ← Ruido blanco (nivel constante)
     |
     |________________________> Frecuencia (Hz)
     0    fc-B/2  fc  fc+B/2
```



---

## 📌 Conclusiones del Paper

1. **Primera formalización teórica**: Se provee una descripción matemática rigurosa de la modulación LoRa (FSCM)

2. **Receptor óptimo eficiente**: Se define un demodulador basado en FFT que es óptimo y de baja complejidad

3. **Superioridad en canales adversos**: Se confirma que FSCM es más robusto que FSK en canales selectivos en frecuencia

4. **Validación para IoT**: Los resultados validan su uso en aplicaciones IoT y LPWAN, donde las condiciones del canal suelen ser adversas

5. **Diversidad inherente**: El barrido de frecuencias proporciona diversidad natural que mejora la robustez

---

## 📚 Referencias 

### Referencia principal:
1. **Vangelista, L. (2017)**. *"Frequency Shift Chirp Modulation: The LoRa Modulation"*. IEEE Signal Processing Letters, Vol. 24, No. 12. DOI: 10.1109/LSP.2017.2762960

