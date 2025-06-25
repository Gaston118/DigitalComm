# Comparación de Desempeño BER/SER de la Modulación FSCM (LoRa) en Canales Planos y Selectivos

## 📜 Contexto: Paper de Referencia

Este trabajo está basado en el artículo:

> **"Frequency Shift Chirp Modulation: The LoRa Modulation"**  
> *Lorenzo Vangelista, IEEE Signal Processing Letters, Vol. 24, No. 12, 2017*  
> [DOI: 10.1109/LSP.2017.2762960](https://ieeexplore.ieee.org/document/8094237)

En este artículo se realiza, por primera vez, una descripción teórica rigurosa del esquema de modulación usado por LoRa: **Frequency Shift Chirp Modulation (FSCM)**. Además, se compara el rendimiento en términos de **BER (Bit Error Rate)** entre FSCM y FSK en canales:

- **AWGN (Flat):** canal aditivo con ruido blanco gaussiano.
- **Selectivo en frecuencia:** canal con multitrayectoria (multipath) que atenúa distintas frecuencias de manera desigual.

---


## 📚 Teoría Clave del Paper

### ✅ 1. Modulación FSCM

Para un símbolo \( s \in \{0, 1, ..., 2^{SF}-1 \} \), la señal transmitida es:

\[
c[n] = \frac{1}{\sqrt{2^{SF}}} \cdot e^{j 2\pi \cdot \left[ \frac{(s + n) \mod 2^{SF}}{2^{SF}} \cdot n \right] }
\]

donde:
- \( SF \): *Spreading Factor* (factor de dispersión, típico entre 7 y 12).
- \( n \): índice de muestra en el intervalo de símbolo.
- Esta fórmula genera una **chirp ascendente con un corrimiento de frecuencia** proporcional al símbolo transmitido.

### ✅ 2. Demodulación óptima

- Se multiplica la señal recibida por un **chirp descendente** (down-chirp), que es el inverso del chirp base.
- Se calcula la **Transformada de Fourier** del resultado.
- El índice del pico de la FFT indica el símbolo transmitido.

### ✅ 3. Modelo de canal selectivo

El canal multipath considerado tiene una respuesta al impulso:

\[
h[n] = \sqrt{0.8} \cdot \delta[n] + \sqrt{0.2} \cdot \delta[n - 1]
\]

Este canal introduce **interferencia intersimbólica (ISI)** y **atenuación selectiva en frecuencia**, lo que afecta negativamente al FSK pero **no tanto al FSCM**, que barre todo el ancho de banda.

---

## 💻 Implementación en Python

### 📐 Parámetros de simulación

| Parámetro           | Valor               |
|---------------------|---------------------|
| SF                  | 7                   |
| Cardinalidad (M)    | 128 símbolos        |
| Ancho de banda (B)  | 125 kHz             |
| Muestras por símbolo| 128                 |
| Nº de símbolos      | 20.000              |
| Nº de bits          | 140.000             |

### ⚙️ Flujo de simulación

1. **Generación de bits aleatorios**.
2. **Codificación LoRa**: los bits se agrupan de a 7 para formar símbolos (enteros).
3. **Modulación FSCM**: se aplica la fórmula del chirp para cada símbolo.
4. **Adición de ruido AWGN** en distintos niveles de SNR (Eb/N0).
5. **Aplicación del canal multipath** (para el caso selectivo).
6. **Demodulación**:
   - Se multiplica por el chirp descendente.
   - Se realiza la FFT.
   - Se detecta el símbolo transmitido como el índice del máximo.
7. **Decodificación a bits**.
8. **Cálculo de tasas de error BER y SER**.
9. **Graficación de curvas**.

---

## 📊 Resultados Obtenidos

El script produce una figura similar a esta:

![Resultados](dc85b09c-c43e-4e3d-b349-47e852ded150.png)

### Observaciones clave:

- En canal plano (AWGN), **BER y SER son similares** para FSCM en comparación con los resultados del paper.
- En canal selectivo en frecuencia:
  - **La BER de FSCM es mucho menor** que la de SER, lo que indica una mayor robustez.
  - Esto **confirma la ventaja teórica de FSCM** sobre FSK mencionada en el paper.
- Las curvas muestran claramente la **superioridad del barrido de espectro (chirp) para combatir la distorsión selectiva en frecuencia**.

---

## 📈 Comparación con el Paper

| Aspecto                      | Paper de Vangelista      | Implementación Propia       |
|------------------------------|---------------------------|-----------------------------|
| Modulación                   | FSCM                      | FSCM                        |
| Canal plano (AWGN)           | ✔️                        | ✔️                          |
| Canal selectivo en frecuencia| ✔️                        | ✔️                          |
| Curvas mostradas             | Solo BER                  | BER y SER                   |
| Rango de SNR                 | -12 dB a -1 dB            | -10 dB a +6 dB              |

---

## 🧪 Extensiones posibles

- ✅ Incluir la modulación **FSK** para comparación directa como en el paper.
- ✅ Simular diferentes **valores de SF** (8, 9, 10, ...).
- ✅ Añadir **canales más complejos**: Rayleigh, Rice, Doppler.
- ✅ Evaluar rendimiento **con y sin codificación** (ej. Hamming, LoRaWAN FEC).
- ✅ Implementar detección **no coherente** o basada en energía.


## 📌 Referencias

1. Vangelista, L. (2017). *Frequency Shift Chirp Modulation: The LoRa Modulation*. IEEE Signal Processing Letters.  
   [IEEE Xplore](https://ieeexplore.ieee.org/document/8094237)
2. LoRa Alliance.  
   [https://lora-alliance.org](https://lora-alliance.org)

---

