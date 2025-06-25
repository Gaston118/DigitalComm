# 📘 Fundamentos Teóricos de la Modulación Frequency Shift Chirp (FSCM)

Este documento resume y explica detalladamente los conceptos teóricos presentados en el artículo:

> **"Frequency Shift Chirp Modulation: The LoRa Modulation"**  
> Lorenzo Vangelista, IEEE Signal Processing Letters, Vol. 24, No. 12, 2017.  
> [DOI: 10.1109/LSP.2017.2762960](https://ieeexplore.ieee.org/document/8094237)

---

## 🔍 Objetivo del Paper

La modulación utilizada por **LoRa**, una tecnología clave en redes **LPWAN** (Low Power Wide Area Networks), ha sido históricamente poco documentada en términos teóricos. El objetivo de este paper es:

- Proporcionar por primera vez una **descripción matemática rigurosa** de la modulación LoRa.
- Definir formalmente la **Frequency Shift Chirp Modulation (FSCM)**.
- Proponer un **demodulador óptimo** de baja complejidad usando FFT.
- Comparar el rendimiento frente a FSK en canales planos y selectivos.

---

## 🧠 Conceptos Teóricos Clave

### 📌 1. Spreading Factor (SF)

- Es un parámetro que determina la **cantidad de bits por símbolo**.
- Denotado por \( SF \in \{7, 8, 9, ..., 12\} \).
- Cada símbolo representa \( M = 2^{SF} \) posibles valores → \( \log_2 M = SF \) bits por símbolo.

---

### 📌 2. Modulación FSCM: Chirps con corrimiento

Para transmitir un símbolo \( s \in \{0, 1, ..., M-1\} \), se genera una señal tipo **chirp**, cuya frecuencia aumenta linealmente con el tiempo, pero con un **corrimiento inicial de frecuencia** dependiente de \( s \).

#### 🔧 Fórmula de la señal transmitida:

\[
c[n] = \frac{1}{\sqrt{M}} \cdot e^{j 2\pi \cdot \frac{(s + n) \bmod M}{M} \cdot n}
\]

Donde:
- \( n \in \{0, 1, ..., M-1\} \) es el índice temporal discreto.
- \( T = \frac{1}{B} \) es el período de muestreo.
- \( M = 2^{SF} \): cantidad total de símbolos posibles.

📌 **Interpretación**:
- Es una señal chirp discreta.
- La información está codificada como un **desplazamiento de frecuencia inicial**, no en el contenido del chirp mismo.
- Se diferencia de los chirps clásicos al ser modulada en frecuencia por la posición \( s \).

---

### 📌 3. Ortogonalidad de las señales chirp

El paper demuestra que las \( M \) señales chirp generadas para cada símbolo son **ortogonales entre sí** en el dominio discreto:

\[
\langle c_i[n], c_q[n] \rangle = 0 \quad \text{para } i \ne q
\]

Esto es esencial para asegurar una detección sin interferencia entre símbolos.

---

### 📌 4. Canal Considerado

#### a) **Canal plano (AWGN):**
- Canal aditivo con ruido blanco gaussiano (modelo ideal).
- No afecta frecuencias de forma selectiva.

#### b) **Canal selectivo en frecuencia:**
\[
h[n] = \sqrt{0.8} \cdot \delta[n] + \sqrt{0.2} \cdot \delta[n-1]
\]

- Representa un **canal multipath** con dos trayectorias.
- Afecta ciertas frecuencias más que otras.
- Genera **desvanecimiento selectivo**, que degrada el desempeño de modulaciones como FSK.

---

### 📌 5. Recepción Óptima de FSCM

#### ✔️ Proceso óptimo de demodulación (conocido como "dechirping"):

1. Multiplicar la señal recibida por un **chirp descendente**:

\[
d[n] = r[n] \cdot e^{-j 2\pi \cdot \frac{n^2}{M}}
\]

2. Calcular la **Transformada Discreta de Fourier (FFT)** de \( d[n] \).
3. El índice del máximo de la FFT corresponde al símbolo transmitido \( \hat{s} \).

---

### 📌 6. Comparación FSCM vs FSK

| Característica          | FSCM (LoRa)                            | FSK tradicional                         |
|-------------------------|----------------------------------------|-----------------------------------------|
| Base matemática         | Chirp con desplazamiento               | Portadoras sinusoidales                 |
| Ortogonalidad           | ✔️ Sí (demostrada en el paper)         | ✔️ Sí (por frecuencia)                  |
| Robustez en multipath   | 🟢 Alta (barre todas las frecuencias)  | 🔴 Baja (puede caer en frecuencias atenuadas) |
| Complejidad demodulación| Baja (uso de FFT)                      | Media (banco de correladores)           |
| Ancho de banda          | Totalmente utilizado                   | Utiliza una porción por símbolo         |

---

## 📊 Resultados del Paper

### Fig. 1 – Curvas de BER no codificada

- En canal **AWGN**, FSK y FSCM tienen desempeño similar.
- En canal **selectivo en frecuencia**, FSCM **supera claramente** a FSK.
- Esto se debe a que FSCM **promedia los efectos del canal** al barrer todo el espectro de frecuencias.

---

## 📌 Conclusiones del Paper

1. Se provee una descripción **teórica rigurosa** de la modulación LoRa (FSCM).
2. Se define un receptor óptimo eficiente basado en **FFT**.
3. Se confirma que **FSCM es más robusto** que FSK en canales no ideales.
4. Esto valida su uso en aplicaciones **IoT y LPWAN**, donde las condiciones del canal suelen ser adversas.

---

## 📚 Referencias

- L. Vangelista, "Frequency Shift Chirp Modulation: The LoRa Modulation", IEEE Signal Processing Letters, 2017.
- N. Benvenuto & G. Cherubini, *Algorithms for Communications Systems and Their Applications*, Wiley, 2002.
- G. Cariolaro, *Unified Signal Theory*, Springer, 2011.

---