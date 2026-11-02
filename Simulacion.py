import numpy as np
import matplotlib.pyplot as plt
import time

def inicializar_particulas(N, L, k, alpha, v_th):
    """
    Inicializa las posiciones y velocidades de N electrones para el Amortiguamiento de Landau.
    Distribución de posiciones: n(x) = n0 * (1 + alpha * cos(k * x))
    Distribución de velocidades: Maxwelliana de media 0 y desviación v_th.
    """
    # 1. Posiciones uniformes perturbadas linealmente
    x = np.linspace(0, L, N, endpoint=False)
    # Corrección de perturbación para muestreo uniforme
    x += (alpha / k) * np.sin(k * x)
    x = np.mod(x, L) # Condiciones de frontera periódicas

    # 2. Velocidades con distribución normal (Maxwelliana)
    v = np.random.normal(0.0, v_th, N)
    
    return x, v

def depositar_carga(x, N, Ng, dx, L):
    """
    Deposita la carga de las partículas a los nodos de la malla (Esquema Cloud-in-Cell / TSC 1D).
    """
    rho = np.zeros(Ng)
    q_part = L / N  # Carga por superpartícula (n0 = 1)

    # Convertir posiciones a coordenadas de malla
    x_grid = x / dx
    i = np.floor(x_grid).astype(int)
    w = x_grid - i

    i_mod = i % Ng
    i_next = (i + 1) % Ng

    # Deposición lineal (CIC)
    np.add.at(rho, i_mod, q_part * (1.0 - w))
    np.add.at(rho, i_next, q_part * w)

    # Densidad de carga neta: (n_ion - n_e), con n_ion = 1.0 uniforme
    rho = 1.0 - (rho / dx)
    return rho

def resolver_poisson_fft(rho, Ng, dx):
    """
    Resuelve la ecuación de Poisson d2phi/dx2 = -rho en el espacio de Fourier (FFT 1D).
    Devuelve el campo eléctrico E(x) en los nodos de la malla.
    """
    # Modos de Fourier
    k_vec = 2.0 * np.pi * np.fft.fftfreq(Ng, d=dx)
    k_vec[0] = 1.0 # Evitar división por cero en k=0 (modo DC)

    rho_hat = np.fft.fft(rho)
    phi_hat = rho_hat / (k_vec**2)
    phi_hat[0] = 0.0 # Potencial promedio nulo

    # Campo eléctrico E = -d(phi)/dx -> en Fourier: E_hat = -i * k * phi_hat
    E_hat = -1j * k_vec * phi_hat
    E = np.real(np.fft.ifft(E_hat))
    return E

def interpolar_campo(x, E_grid, dx, Ng):
    """
    Interpola el campo eléctrico de los nodos de la malla a la posición exacta de cada partícula.
    """
    x_grid = x / dx
    i = np.floor(x_grid).astype(int)
    w = x_grid - i

    i_mod = i % Ng
    i_next = (i + 1) % Ng

    E_part = E_grid[i_mod] * (1.0 - w) + E_grid[i_next] * w
    return E_part

def simular_landau_serial(N=500000, Ng=64, k=0.5, alpha=0.05, dt=0.1, n_steps=300):
    """
    Simulación principal en versión serial.
    """
    L = 2.0 * np.pi / k
    dx = L / Ng
    v_th = 1.0 # Velocidad térmica normalizada
    qm = -1.0  # Relación carga/masa del electrón normalizada

    print(f"Iniciando simulación serial con N = {N} partículas, Ng = {Ng} nodos...")
    
    # 1. Inicialización
    x, v = inicializar_particulas(N, L, k, alpha, v_th)
    
    # Avance previo medio paso de velocidad (Leapfrog)
    rho = depositar_carga(x, N, Ng, dx, L)
    E_grid = resolver_poisson_fft(rho, Ng, dx)
    E_part = interpolar_campo(x, E_grid, dx, Ng)
    v -= 0.5 * qm * E_part * dt

    energia_electrica = np.zeros(n_steps)
    tiempos = np.zeros(n_steps)

    t_inicio = time.perf_counter()

    # 2. Bucle temporal principal
    for step in range(n_steps):
        # Actualizar posiciones: x(t+dt) = x(t) + v(t+dt/2)*dt
        x += v * dt
        x = np.mod(x, L) # Frontera periódica

        # Depositar carga y resolver campo eléctrico
        rho = depositar_carga(x, N, Ng, dx, L)
        E_grid = resolver_poisson_fft(rho, Ng, dx)

        # Actualizar velocidades: v(t+3/2 dt) = v(t+1/2 dt) + q/m * E * dt
        E_part = interpolar_campo(x, E_grid, dx, Ng)
        v += qm * E_part * dt

        # Registrar la energía electrostática total U_E = 0.5 * sum(E^2) * dx
        energia_electrica[step] = 0.5 * np.sum(E_grid**2) * dx
        tiempos[step] = step * dt

    t_fin = time.perf_counter()
    print(f"Simulación completada en {t_fin - t_inicio:.3f} segundos.")

    return tiempos, energia_electrica

if __name__ == "__main__":
    t, U_E = simular_landau_serial(N=500000, Ng=64, k=0.5, alpha=0.05, dt=0.1, n_steps=300)

    # Graficar la energía electrostática (debe decrecer exponencialmente)
    plt.figure(figsize=(8, 5))
    plt.semilogy(t, U_E, label=r'Energía Electrostática $U_E(t)$', color='navy')
    plt.xlabel('Tiempo ($\omega_{pe}^{-1}$)')
    plt.ylabel(r'Energía Electrostática $\int E^2 dx$')
    plt.title('Amortiguamiento de Landau 1D (Simulación PIC Serial)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig('landau_damping_serial.png', dpi=300)
    plt.show()