# Modulación FSCM (LoRa) en Canales Planos y Selectivos

## 📜 Contexto: Paper de Referencia

Este trabajo está basado en el artículo:

> **"Frequency Shift Chirp Modulation: The LoRa Modulation"**  
> *Lorenzo Vangelista, IEEE Signal Processing Letters, Vol. 24, No. 12, 2017*  

En este artículo se realiza, por primera vez, una descripción teórica rigurosa del esquema de modulación usado por LoRa: **Frequency Shift Chirp Modulation (FSCM)**. Además, se compara el rendimiento en términos de **BER (Bit Error Rate)** entre FSCM y FSK en canales:

- **AWGN (Flat):** canal aditivo con ruido blanco gaussiano.
- **Selectivo en frecuencia:** canal con multitrayectoria (multipath) que atenúa distintas frecuencias de manera desigual.

---

## 🔍 Objetivo del Paper

La modulación utilizada por **LoRa**, una tecnología clave en redes **LPWAN** (Low Power Wide Area Networks), ha sido históricamente poco documentada en términos teóricos. Los objetivos principales del paper son:

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

```
c[n] = (1/√M) · exp(j·2π·[(s + n) mod M]/M · n)
```

Donde:
- **n** ∈ {0, 1, ..., M-1} es el índice temporal discreto
- **T** = 1/B es el período de muestreo
- **M** = 2^SF: cantidad total de símbolos posibles
- **s**: símbolo a transmitir

#### 📌 Interpretación física:
- Es una **señal chirp discreta** con corrimiento de frecuencia
- La información está codificada como un **desplazamiento de frecuencia inicial**
- Se diferencia de los chirps clásicos al ser modulada en frecuencia por la posición s
- El chirp "barre" todo el ancho de banda disponible

### ✅ 3. Ortogonalidad de las Señales Chirp

El paper demuestra matemáticamente que las M señales chirp generadas para cada símbolo son **ortogonales entre sí** en el dominio discreto:

```
⟨c_i[n], c_q[n]⟩ = 0    para i ≠ q
```

**Importancia**: Esta ortogonalidad es esencial para asegurar una detección sin interferencia entre símbolos y permite la separación perfecta en condiciones ideales.

### ✅ 4. Demodulación Óptima ("Dechirping")

El proceso óptimo de demodulación, conocido como **"dechirping"**, consiste en:

#### Paso 1: Multiplicación por chirp descendente
```
d[n] = r[n] · exp(-j·2π·n²/M)
```

#### Paso 2: Transformada de Fourier
- Se calcula la **FFT** de d[n]
- El resultado es: **FFT{d[n]}**

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

### ⚙️ Flujo de simulación detallado

#### 1. **Generación de datos**
```python
# Generación de bits aleatorios
bits = np.random.randint(0, 2, num_bits)
```

#### 2. **Codificación LoRa**
```python
# Agrupación de bits en símbolos
symbols = []
for i in range(0, len(bits), SF):
    symbol = sum(bits[i+j] * (2**j) for j in range(SF))
    symbols.append(symbol)
```

#### 3. **Modulación FSCM**
```python
# Generación de señales chirp para cada símbolo
def generate_chirp(symbol, SF):
    M = 2**SF
    n = np.arange(M)
    return (1/np.sqrt(M)) * np.exp(1j * 2 * np.pi * ((symbol + n) % M) / M * n)
```

#### 4. **Canal de transmisión**
- **Canal AWGN**: Adición de ruido gaussiano complejo
- **Canal selectivo**: Convolución con respuesta al impulso h[n]

#### 5. **Demodulación FSCM**
```python
def demodulate_fscm(received_signal, SF):
    M = 2**SF
    n = np.arange(M)
    # Dechirping
    dechirped = received_signal * np.exp(-1j * 2 * np.pi * n**2 / M)
    # FFT
    fft_result = np.fft.fft(dechirped)
    # Detección
    detected_symbol = np.argmax(np.abs(fft_result))
    return detected_symbol
```

---

## 📊 Resultados y Análisis

### Curvas de rendimiento esperadas:

**Configuración típica**:
- **Eje X**: Eb/N0 (dB) - Relación energía por bit a densidad de ruido
- **Eje Y**: BER/SER (escala logarítmica)
- **Rango**: -10 dB a +6 dB

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

| Característica              | FSCM (LoRa)                           | FSK Tradicional                       |
|-----------------------------|---------------------------------------|---------------------------------------|
| **Base matemática**         | Chirp con desplazamiento             | Portadoras sinusoidales               |
| **Ortogonalidad**           | ✔️ Sí (demostrada en el paper)        | ✔️ Sí (por frecuencia)                |
| **Robustez en multipath**   | 🟢 Alta (barre todas las frecuencias) | 🔴 Baja (puede caer en frecuencias atenuadas) |
| **Complejidad demodulación**| Baja (uso de FFT)                     | Media (banco de correladores)         |
| **Ancho de banda**          | Totalmente utilizado                  | Utiliza una porción por símbolo       |
| **Diversidad en frecuencia**| ✔️ Inherente                          | ❌ No disponible                      |
| **Sincronización**          | Menos crítica                         | Más crítica                           |

---

## 🔬 Fundamentos Matemáticos Avanzados

### Señal Chirp Base

La señal chirp fundamental utilizada en LoRa puede expresarse como:

```
c₀[n] = exp(j·π·n²/M)  para n = 0, 1, ..., M-1
```

### Modulación por Desplazamiento Circular

Para transmitir un símbolo s, se aplica un desplazamiento circular:

```
cₛ[n] = c₀[(n + s) mod M]
```

### Propiedades de Correlación

La correlación cruzada entre diferentes símbolos FSCM satisface:

```
R_ij = (1/M) · Σ(n=0 to M-1) cᵢ[n] · cⱼ*[n] = δᵢⱼ
```

donde δᵢⱼ es la delta de Kronecker.

### Respuesta en Frecuencia

El espectro de potencia de una señal FSCM es aproximadamente plano sobre el ancho de banda B, lo que proporciona diversidad en frecuencia natural.

---

## 🧪 Extensiones y Mejoras Posibles

### Análisis adicionales del paper:
- ✅ **Codificación de canal**: Hamming, Reed-Solomon, LoRaWAN FEC
- ✅ **Diferentes valores de SF**: Análisis de 7 a 12
- ✅ **Canales más complejos**: Rayleigh, Rice, con efecto Doppler
- ✅ **Detección no coherente**: Basada en energía vs coherente

### Optimizaciones propuestas:
- ✅ **Sincronización**: Estimación de offset de frecuencia y tiempo
- ✅ **Ecualización**: Técnicas para canales selectivos más complejos
- ✅ **Redes multi-usuario**: Análisis de interferencia y capacidad
- ✅ **Implementación en tiempo real**: GNU Radio, USRP

---

## 🔍 Análisis de Complejidad Computacional

### Modulación FSCM:
- **Operaciones por símbolo**: O(M) multiplicaciones complejas
- **Memoria requerida**: O(M) muestras complejas
- **Implementación**: Tabla de lookup para exponenciales

### Demodulación FSCM:
- **Dechirping**: O(M) multiplicaciones complejas
- **FFT**: O(M log M) operaciones
- **Detección**: O(M) comparaciones
- **Total**: O(M log M) por símbolo

### Comparación con FSK:
- **FSK modulación**: O(1) por símbolo
- **FSK demodulación**: O(M) correlaciones → O(M²) total
- **Ventaja FSCM**: Escalabilidad logarítmica vs cuadrática

---

## 📌 Conclusiones del Paper Original

1. **Primera formalización teórica**: Se provee una descripción matemática rigurosa de la modulación LoRa (FSCM)

2. **Receptor óptimo eficiente**: Se define un demodulador basado en FFT que es óptimo y de baja complejidad

3. **Superioridad en canales adversos**: Se confirma que FSCM es más robusto que FSK en canales selectivos en frecuencia

4. **Validación para IoT**: Los resultados validan su uso en aplicaciones IoT y LPWAN, donde las condiciones del canal suelen ser adversas

5. **Diversidad inherente**: El barrido de frecuencias proporciona diversidad natural que mejora la robustez

---

## 🎯 Aplicaciones Prácticas

### Tecnologías que usan FSCM:
- **LoRaWAN**: Redes de área amplia de baja potencia
- **Semtech LoRa**: Chips de comunicación IoT
- **Aplicaciones IoT**: Sensores, medidores inteligentes, agricultura
- **Comunicaciones satelitales**: Adaptación para enlaces de larga distancia

### Ventajas en implementaciones reales:
- **Bajo consumo de energía**: Eficiencia en dispositivos battery-powered
- **Largo alcance**: Comunicación de varios kilómetros
- **Penetración en edificios**: Robustez en entornos urbanos
- **Tolerancia a interferencia**: Coexistencia con otras tecnologías

---

## 📚 Referencias 

### Referencia principal:
1. **Vangelista, L. (2017)**. *"Frequency Shift Chirp Modulation: The LoRa Modulation"*. IEEE Signal Processing Letters, Vol. 24, No. 12. DOI: 10.1109/LSP.2017.2762960

---

## 💡 Resumen Ejecutivo

La modulación **Frequency Shift Chirp Modulation (FSCM)** utilizada en LoRa representa un avance significativo en las comunicaciones digitales para aplicaciones IoT. El paper de Vangelista proporciona por primera vez una base teórica sólida que explica por qué LoRa es tan efectivo en condiciones adversas:

**Factores clave de éxito**:
1. **Diversidad en frecuencia inherente** mediante el barrido completo del espectro
2. **Ortogonalidad matemática** que permite detección sin interferencia
3. **Demodulación eficiente** usando FFT para baja complejidad computacional
4. **Robustez ante canales selectivos** que degradan otras modulaciones

Estas características fundamentales hacen de FSCM la elección ideal para el ecosistema IoT moderno, donde se requiere comunicación confiable de largo alcance con dispositivos de baja potencia en entornos desafiantes.
