import numpy as np
import matplotlib.pyplot as plt
from math import erfc


# --- PASO 1: Generar bits aleatorios ---

np.random.seed(42)           # Inicio para reproducir entre los scripts
M = 8                        # Orden de modulación
k = int(np.log2(M))          # Bits por segundo  
num_simbolos = 100_000       # Cantidad de símbolos
num_bits = num_simbolos * k  # Total de bits = 3000,000 bits

# Generación de la fuente binario aleatoria equiprobable(0s y 1s)
bits_tx = np.random.randint(0, 2, size=num_bits)

print("Resultado: 1")
print(f"Total de bits generados: {num_bits}")
print(f"Primeros 15 bits transmitidos: {bits_tx[:15]}")



# --- PASO 2: Convertir bits a símbolos ---
# Reorganizar los bits en bloques de 3 columnas (100,000 x 3)
bits_agrupados = bits_tx.reshape((num_simbolos, k))

# Convertir cada triplete de bits a su entero decimal (0..7)
valores_decimales = bits_agrupados[:, 0] * 4 + bits_agrupados[:, 1] * 2 + bits_agrupados[:, 2]

# Tabla de mapeo Binario Natural -> Índice de Símbolo Gray
tabla_bin_a_gray = np.array([0, 1, 3, 2, 7, 6, 4, 5])
simbolos_tx = tabla_bin_a_gray[valores_decimales]

print("Resultado: 2")
print("Primeros 5 tripletes de bits:\n", bits_agrupados[:5])
print("Índices de símbolos Gray asignados:", simbolos_tx[:5])


# --- PASO 3: Modular con 8-PSK (mapeo Gray) ---
# Generar los 8 puntos ideales de la constelación 8-PSK normalizada (Es = 1)
constelacion_8psk = np.exp(1j * 2 * np.pi * np.arange(M) / M)

# Mapear cada símbolo transmitido a su punto complejo correspondiente
senal_tx = constelacion_8psk[simbolos_tx]

energia_promedio = np.mean(np.abs(senal_tx) ** 2)
print("Resultado: 3")
print(f"Energía promedio por símbolo: {energia_promedio:.2f}")
print("Primeros 5 símbolos:\n", np.round(senal_tx[:5], 4))


# --- PASO 4: Constelación de la señal transmitida ---
plt.figure(figsize=(6.5, 6.5))
# Graficar una muestra de 1,000 símbolos transmitidos
plt.scatter(np.real(senal_tx[:1000]), np.imag(senal_tx[:1000]),
            color='#0055A4', s=80, label='Símbolos transmitidos (sin ruido)', zorder=3)

# Dibujar círculo unitario de referencia (Es = 1)
angulo = np.linspace(0, 2 * np.pi, 300)
plt.plot(np.cos(angulo), np.sin(angulo), '--', color='gray', alpha=0.6, label='Círculo unitario ($E_s=1$)')

# Etiquetar cada punto de la constelación con su triplete Gray
etiquetas_gray = ['000', '001', '011', '010', '110', '111', '101', '100']
for idx, punto in enumerate(constelacion_8psk):
    plt.annotate(f"m={idx}\n({etiquetas_gray[idx]})",
                 (np.real(punto) * 1.16, np.imag(punto) * 1.16),
                ha='center', va='center', fontsize=9, fontweight='bold')

plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.xlim(-1.45, 1.45)
plt.ylim(-1.45, 1.45)
plt.grid(True, linestyle=':', alpha=0.7)
plt.title('Diagrama de Constelación 8-PSK (Señal Transmitida)')
plt.xlabel('Componente en Fase (I)')
plt.ylabel('Componente en Cuadratura (Q)')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
print("Resultado: 4-Diagrama de Constelación 8-PSK")


# --- PASO 5: Añadir ruido AWGN  ---
def canal_awgn(senal, snr_db):
    """Agrega ruido blanco gaussiano complejo (AWGN) a la señal según el SNR en dB."""
    snr_lineal = 10 ** (snr_db / 10.0)
    energia_simbolo = np.mean(np.abs(senal) ** 2)   # Es = 1.0
    n0 = energia_simbolo / snr_lineal
    desviacion_std = np.sqrt(n0 / 2.0)

    ruido = desviacion_std * (np.random.randn(len(senal)) + 1j * np.random.randn(len(senal)))
    return senal + ruido

# Ejemplo de paso por el canal con SNR = 5 dB
senal_rx_5db = canal_awgn(senal_tx, snr_db=5)
print("Resultado: 5")
print("Muestra de los primeros 5 símbolos recibidos (SNR = 5 dB):\n", np.round(senal_rx_5db[:5], 4))



# --- PASO 6: Demodular (detector de mínima distancia)  ---
def demodular_8psk(senal_rx, constelacion):
    """Demodula símbolos 8-PSK mediante criterio de mínima distancia euclidiana."""
    # Matriz de distancias de tamaño (num_simbolos, 8)
    distancias = np.abs(senal_rx[:, None] - constelacion[None, :])
    return np.argmin(distancias, axis=1)

simbolos_rx_5db = demodular_8psk(senal_rx_5db, constelacion_8psk)
print("Resultado: 6")
print("Símbolos transmitidos (primeros 10):", simbolos_tx[:10])
print("Símbolos demodulados  (primeros 10):", simbolos_rx_5db[:10])


# --- PASO 7: Constelación recibida (SNR = 0, 5 y 10 dB)  ---
niveles_snr = [0, 5, 10]
muestra_plot = 2000  # Cantidad de puntos a graficar para visualización clara

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, snr in zip(axes, niveles_snr):
    rx_temp = canal_awgn(senal_tx[:muestra_plot], snr_db=snr)

    ax.scatter(np.real(rx_temp), np.imag(rx_temp),
               color='#2E86AB', alpha=0.35, s=12, label='Símbolos recibidos')
    ax.scatter(np.real(constelacion_8psk), np.imag(constelacion_8psk),
               color='#D90429', marker='x', s=90, linewidths=2, label='Constelación ideal')

    ax.axhline(0, color='black', linewidth=0.7)
    ax.axvline(0, color='black', linewidth=0.7)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_title(f'Constelación Recibida (SNR = {snr} dB)')
    ax.set_xlabel('Componente en Fase (I)')
    ax.set_ylabel('Componente en Cuadratura (Q)')
    ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.show()
print("Resultado: 7-Constelación Recibida o,5,10 dB")


# --- PASO 8: Transformar símbolos a bits  ---
# Tabla inversa: Índice de Símbolo Gray (0..7) -> Entero Binario Natural (0..7)
tabla_gray_a_bin = np.array([0, 1, 3, 2, 6, 7, 5, 4])

def simbolos_a_bits(simbolos_rx, k=3):
    """Convierte índices de símbolos Gray demodulados nuevamente a flujo de bits."""
    decimales_rx = tabla_gray_a_bin[simbolos_rx]
    # Extraer los 3 bits (MSB a LSB) por cada símbolo
    bits_matriz = ((decimales_rx[:, None] >> np.arange(k - 1, -1, -1)) & 1)
    return bits_matriz.flatten()

bits_rx_5db = simbolos_a_bits(simbolos_rx_5db, k=k)
print("Resultado 8:")
print("Primeros 15 bits transmitidos:", bits_tx[:15])
print("Primeros 15 bits recibidos   :", bits_rx_5db[:15])


# --- PASO 9: BER vs Eb/N0 (Montecarlo)  ---
from math import erfc

eb_n0_db_rango = np.arange(0, 11, 1)  # Valores de Eb/N0 de 0 a 10 dB
ber_simulado = []
ber_teorico = []

for eb_n0_db in eb_n0_db_rango:
    # Convertir Eb/N0 (dB) a SNR por símbolo (dB): SNR = Eb/N0 + 10*log10(k)
    snr_db = eb_n0_db + 10 * np.log10(k)

    # Transmisión por canal AWGN, demodulación y recuperación de bits
    senal_rx = canal_awgn(senal_tx, snr_db)
    simbolos_rx = demodular_8psk(senal_rx, constelacion_8psk)
    bits_rx = simbolos_a_bits(simbolos_rx, k=k)

    # Cálculo del BER simulado
    errores = np.sum(bits_tx != bits_rx)
    ber = errores / num_bits
    ber_simulado.append(ber)

    # Cálculo del BER teórico aproximado para 8-PSK Gray
    eb_n0_lin = 10 ** (eb_n0_db / 10.0)
    arg_q = np.sqrt(6 * eb_n0_lin) * np.sin(np.pi / 8)
    prob_bit_teorica = (2 / 3) * (0.5 * erfc(arg_q / np.sqrt(2)))
    ber_teorico.append(prob_bit_teorica)

    #print("Resultado 9")
    print(f"Eb/N0 = {eb_n0_db:2d} dB | Errores = {errores:6d} | BER Simulado = {ber:.6f} | BER Teórico = {prob_bit_teorica:.6f}")

# Gráfica semilogarítmica de desempeño BER vs Eb/N0
plt.figure(figsize=(8, 5.5))
plt.semilogy(eb_n0_db_rango, ber_simulado, 'o-', color='#0055A4', linewidth=2, label='8-PSK Simulado (Montecarlo)')
plt.semilogy(eb_n0_db_rango, ber_teorico, 's--', color='#D90429', linewidth=1.5, label='8-PSK Teórico (Aprox. Gray)')
plt.axhline(1e-3, color='green', linestyle=':', linewidth=1.2, label='Umbral referencia voz ($BER = 10^{-3}$)')

plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.title('Curva de Desempeño BER vs $E_b/N_0$ para Modulación 8-PSK en Canal AWGN')
plt.xlabel('$E_b / N_0$ (dB)')
plt.ylabel('Tasa de Error de Bit (BER)')
plt.ylim(1e-4, 1)
plt.legend()
plt.tight_layout()
plt.show()

