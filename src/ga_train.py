import os
import numpy as np
import gymnasium as gym
from config import CONFIGURACIONES 

# ------------------- Configuración -------------------
SEED = 0
np.random.seed(SEED)

RUTA_MEJOR = "models/best_theta.npz"

ACCIONES = 3
CARACT = 6                # [x, v, 1, x^2, v^2, x*v]


# ------------------- Representación -------------------
def caracteristicas(obs: np.ndarray) -> np.ndarray:
    #Convierte la observación [x, v] en φ = [x, v, 1, x^2, v^2, x*v].
    x, v = obs
    return np.array([x, v, 1.0, x * x, v * v, x * v], dtype=np.float32)


def accion(obs: np.ndarray, theta: np.ndarray) -> int:
    """
    Calcula la acción como argmax_a W_a · φ(obs).
    theta: vector de 18 -> W de (3,6)
    """
    W = theta.reshape(ACCIONES, CARACT)    # (3,6)
    phi = caracteristicas(obs)             # (6,)
    q = W @ phi                            # (3,)
    return int(np.argmax(q))               # 0, 1, 2


# ------------------- Evaluación -------------------
def evaluar_theta(theta: np.ndarray, EPISODIOS_EVAL: int, PASOS_MAX: int, seed_offset: int = 0):
    """
    Evalúa un individuo en EPISODIOS_EVAL episodios.
    Retorna:
      fitness (float), recompensa_prom (float), x_max_prom (float)
    """
    env = gym.make("MountainCar-v0")  # sin render para que sea rápido
    recompensas = []
    max_positions = []

    for i in range(EPISODIOS_EVAL):
        obs, info = env.reset(seed=seed_offset + i)
        ret = 0.0
        x_max = -np.inf

        for step in range(PASOS_MAX):
            a = accion(obs, theta)
            obs, r, terminated, truncated, info = env.step(a)
            ret += r
            x_max = max(x_max, obs[0])     # obs[0] es la posición x
            if terminated or truncated:
                break

        recompensas.append(ret)
        max_positions.append(x_max)

    env.close()

    r_prom = float(np.mean(recompensas))
    x_prom = float(np.mean(max_positions))

    # Mezcla recompensa (menos pasos = mejor) + empuje a x=0.5
    fitness = r_prom + 100.0 * x_prom
    return float(fitness), r_prom, x_prom


# ------------------- Operadores GA -------------------
def torneo(poblacion: np.ndarray, fitnesses: list[float], k: int = 3) -> np.ndarray:
    #Selección por torneo: devuelve una copia del ganador.
    idxs = np.random.choice(len(poblacion), k, replace=False)
    ganador = idxs[np.argmax([fitnesses[i] for i in idxs])]
    return poblacion[ganador].copy()

#Cruce uniforme gen a gen
def cruce(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    mascara = np.random.rand(p1.size) < 0.5
    hijo = np.where(mascara, p1, p2)
    return hijo

#Mutación gaussiana por gen con prob. TASA_MUTACION
def mutar(hijo: np.ndarray, TASA_MUTACION: float, DESV_MUTACION: float) -> np.ndarray:
    mascara = np.random.rand(hijo.size) < TASA_MUTACION
    ruido = np.random.randn(hijo.size) * DESV_MUTACION
    return hijo + mascara * ruido


# ------------------- Bucle principal -------------------
def ejecutarConfiguraciones(config_name, config_params):
    print(f"\n{'='*60}")
    print(f"EJECUTANDO CONFIGURACIÓN: {config_name}")
    print(f"{'='*60}")
    
    # Extraer parámetros de la configuración
    POBLACION = config_params['POBLACION']
    GENERACIONES = config_params['GENERACIONES']
    ELITE = config_params['ELITE']
    TASA_MUTACION = config_params['TASA_MUTACION']
    DESV_MUTACION = config_params['DESV_MUTACION']
    EPISODIOS_EVAL = config_params['EPISODIOS_EVAL']
    PASOS_MAX = config_params['PASOS_MAX']
    
    RUTA_MEJOR = f"models/best_theta_{config_name}.npz"

    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    historial = {
        'fitness_max': [],
        'fitness_prom': [],
        'fitness_min': [],
        'recompensa_prom': [],
        'xmax_prom': [],
        'config_name': config_name,
        'parametros': config_params
    }


    # Cromosoma: 3 acciones x 6 características = 18 genes
    poblacion = np.random.randn(POBLACION, ACCIONES * CARACT) * 0.5

    mejor_theta = None
    mejor_fitness_global = -1e9

    for g in range(GENERACIONES):
        fitnesses = []
        recomp_prom = []
        xmax_prom = []

        # Evaluar población
        for i, theta in enumerate(poblacion):
            fit, r, xm = evaluar_theta(theta, seed_offset=g * 1000 + i * 10, EPISODIOS_EVAL=EPISODIOS_EVAL, PASOS_MAX=PASOS_MAX)
            fitnesses.append(fit)
            recomp_prom.append(r)
            xmax_prom.append(xm)

        # Estadísticas de generación
        i_best_gen = int(np.argmax(fitnesses))
        fit_best_gen = float(fitnesses[i_best_gen])
        fit_prom_gen = float(np.mean(fitnesses))
        fit_min_gen = float(np.min(fitnesses))

        # Guardar en historial
        historial['fitness_max'].append(fit_best_gen)
        historial['fitness_prom'].append(fit_prom_gen)
        historial['fitness_min'].append(fit_min_gen)
        historial['recompensa_prom'].append(np.mean(recomp_prom))
        historial['xmax_prom'].append(np.mean(xmax_prom))

        if fit_best_gen > mejor_fitness_global:
            mejor_fitness_global = fit_best_gen
            mejor_theta = poblacion[i_best_gen].copy()
            np.savez(RUTA_MEJOR, theta=mejor_theta)

        print(
            f"G{g:02d} | fit best: {fit_best_gen:7.2f} | "
            f"fit avg: {np.mean(fitnesses):7.2f} | "
            f"fit min: {fit_min_gen:7.2f} | "
            f"rew avg: {np.mean(recomp_prom):6.2f} | "
            f"x_max: {np.mean(xmax_prom):.3f}"
        )

        # Nueva población con elitismo
        orden = np.argsort(fitnesses)[::-1]
        nueva_poblacion = [poblacion[i].copy() for i in orden[:ELITE]]

        # Resto: torneo + cruce + mutación
        while len(nueva_poblacion) < POBLACION:
            p1 = torneo(poblacion, fitnesses, k=3)
            p2 = torneo(poblacion, fitnesses, k=3)
            hijo = cruce(p1, p2)
            hijo = mutar(hijo, TASA_MUTACION, DESV_MUTACION)
            nueva_poblacion.append(hijo)

        poblacion = np.stack(nueva_poblacion, axis=0)

    np.savez(f"data/historial_{config_name}.npz", **historial)
    print(f"\nDatos guardados en: data/historial_{config_name}.npz")
    print(f"Mejor fitness global: {mejor_fitness_global:.2f}")

    return historial, mejor_theta

def main():
    resultados = {}


    for config_name, config_params in CONFIGURACIONES.items():

        historial, mejor_theta = ejecutarConfiguraciones(config_name, config_params)
        resultados[config_name] = {
            'historial': historial,
            'mejor_theta': mejor_theta
        }

        print(f"\n{'='*60}")
        print(f"CONFIGURACIÓN {config_name} COMPLETADA")
        print(f"{'='*60}\n")

    return resultados

if __name__ == "__main__":
    resultados = main()