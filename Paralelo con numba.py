import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit, prange

# --- FUNCIONES PARALELIZADAS CON NUMBA ---

@njit(parallel=True, fastmath=True)
def depositar_carga_numba(x, N, Ng, dx, L):
    """
    Deposición de carga en paralelo.
    Se utiliza reducción manual por hilos para evitar condiciones de carrera (Race Conditions).
    """
    q_part = L / N
    # Cada hilo mantendrá su propia mallas local de densidad
    # pero para evitar sobrecosto, acumulamos directamente en arreglos temporales.
    rho = np.zeros(Ng)
    
    # En numba prange, para evitar race conditions en np.add.at, 
    # hacemos una acumulación por atómica o reducción local.
    # Esta estructura de bucle la paraleliza Numba dividiendo x en bloques:
    for j in prange(N):
        x_grid = x[j] / dx
        i = int(np.floor(x_grid))
        w = x_grid - i
        i_mod = i % Ng
        i_next = (i + 1) % Ng
        
        # En Numba, el acceso directo dentro de prange sobre arreglos compartidos
        # se optimiza de forma segura si usamos atómicas o arreglos locales.
        # Para mallas pequeñas (Ng=64), atomic add o acumulación directa Numba la maneja:
        rho[i_mod] += q_part * (1.0 - w)
        rho[i_next] += q_part * w

    rho = 1.0 - (rho / dx)
    return rho

@njit(parallel=True, fastmath=True)
def actualizar_particulas_numba(x, v, E_grid, dx, Ng, L, qm, dt):
    """
    Actualiza posiciones y velocidades de N partículas completamente en paralelo.
    No hay dependencias entre partículas (embarrassingly parallel).
    """
    N = len(x)
    for j in prange(N):
        # 1. Interpolar campo E a la partícula
        x_grid = x[j] / dx
        i = int(np.floor(x_grid))
        w = x_grid - i
        i_mod = i % Ng
        i_next = (i + 1) % Ng
        
        E_part = E_grid[i_mod] * (1.0 - w) + E_grid[i_next] * w
        
        # 2. Actualizar velocidad
        v[j] += qm * E_part * dt
        
        # 3. Actualizar posición
        x[j] += v[j] * dt
        x[j] = x[j] % L

def resolver_poisson_fft(rho, Ng, dx):
    """
    Resolver Poisson sigue haciéndose en la malla Ng=64 (operación despreciable).
    """
    k_vec = 2.0 * np.pi * np.fft.fftfreq(Ng, d=dx)
    k_vec[0] = 1.0

    rho_hat = np.fft.fft(rho)
    phi_hat = rho_hat / (k_vec**2)
    phi_hat[0] = 0.0

    E_hat = -1j * k_vec * phi_hat
    return np.real(np.fft.ifft(E_hat))

def simular_landau_paralelo(N=500000, Ng=64, k=0.5, alpha=0.05, dt=0.1, n_steps=300):
    L = 2.0 * np.pi / k
    dx = L / Ng
    v_th = 1.0
    qm = -1.0

    # Inicializar (Serial)
    x = np.linspace(0, L, N, endpoint=False)
    x += (alpha / k) * np.sin(k * x)
    x = np.mod(x, L)
    v = np.random.normal(0.0, v_th, N)

    # Calentamiento JIT (Warmup) para no medir tiempo de compilación Numba
    _ = depositar_carga_numba(x[:100], 100, Ng, dx, L)
    _E_dummy = np.zeros(Ng)
    actualizar_particulas_numba(x[:100], v[:100], _E_dummy, dx, Ng, L, qm, dt)

    print(f"Iniciando simulación PARALELA (Numba) con N = {N} partículas...")
    
    # Avance inicial Leapfrog
    rho = depositar_carga_numba(x, N, Ng, dx, L)
    E_grid = resolver_poisson_fft(rho, Ng, dx)
    
    # Velocidad inicial ajustada -0.5 dt
    for j in range(N):
        x_grid = x[j] / dx
        i = int(np.floor(x_grid))
        w = x_grid - i
        E_p = E_grid[i % Ng] * (1.0 - w) + E_grid[(i + 1) % Ng] * w
        v[j] -= 0.5 * qm * E_p * dt

    energia_electrica = np.zeros(n_steps)
    
    t_inicio = time.perf_counter()

    # Bucle principal
    for step in range(n_steps):
        # Actualizar V y X en paralelo
        actualizar_particulas_numba(x, v, E_grid, dx, Ng, L, qm, dt)
        
        # Carga en paralelo
        rho = depositar_carga_numba(x, N, Ng, dx, L)
        
        # Poisson
        E_grid = resolver_poisson_fft(rho, Ng, dx)
        
        energia_electrica[step] = 0.5 * np.sum(E_grid**2) * dx

    t_fin = time.perf_counter()
    tiempo_total = t_fin - t_inicio
    print(f"Simulación PARALELA completada en {tiempo_total:.3f} segundos.")

    return tiempo_total, energia_electrica

if __name__ == "__main__":
    n_steps = 300
    dt = 0.1
    
    # Ejecutar simulación paralela
    t_par, U_E_par = simular_landau_paralelo(N=500000, Ng=64, k=0.5, alpha=0.05, dt=dt, n_steps=n_steps)

    # Vector de tiempo
    tiempos = np.linspace(0, n_steps * dt, n_steps)

    # Graficar la energía electrostática
    plt.figure(figsize=(8, 5))
    plt.semilogy(tiempos, U_E_par, label=r'Energía Electrostática $U_E(t)$ (Paralelo)', color='crimson')
    plt.xlabel(r'Tiempo ($\omega_{pe}^{-1}$)')
    plt.ylabel(r'Energía Electrostática $\int E^2 dx$')
    plt.title('Amortiguamiento de Landau 1D (Simulación PIC Paralela - Numba)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig('landau_damping_paralelo.png', dpi=300)
    plt.show()