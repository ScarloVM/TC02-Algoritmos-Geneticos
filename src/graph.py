import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.ndimage import gaussian_filter1d

def cargar_datos(config_name=None):
    """Carga los datos de entrenamiento guardados"""
    if config_name:
        archivo = f"data/historial_{config_name}.npz"
        datos = np.load(archivo, allow_pickle=True)
        return {key: datos[key] for key in datos.files}
    else:
        # Cargar todas las configuraciones
        archivos = glob.glob("data/historial_*.npz")
        resultados = {}
        for archivo in archivos:
            config_name = archivo.split('_')[-1].replace('.npz', '')
            datos = np.load(archivo, allow_pickle=True)
            resultados[config_name] = {key: datos[key] for key in datos.files}
        return resultados

def graficar_progreso(config_name, datos):
    """Grafica el progreso para una configuración específica"""
    generaciones = range(len(datos['fitness_max']))
    
    plt.figure(figsize=(12, 6))
    
    # Graficar fitness máximo, promedio y mínimo
    plt.plot(generaciones, datos['fitness_max'], 'b-', label='Fitness Máximo', linewidth=2)
    plt.plot(generaciones, datos['fitness_prom'], 'r-', label='Fitness Promedio', linewidth=2)
    plt.plot(generaciones, datos['fitness_min'], 'g-', label='Fitness Mínimo', alpha=0.7)
    
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.title(f'Progreso del Fitness - Configuración: {config_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("graficas", exist_ok=True)
    plt.savefig(f"graficas/progreso_fitness_{config_name}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Gráfica de métricas adicionales
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(generaciones, datos['recompensa_prom'], 'orange', label='Recompensa Promedio', linewidth=2)
    plt.ylabel('Recompensa')
    plt.title(f'Otras Métricas - Configuración: {config_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(generaciones, datos['xmax_prom'], 'purple', label='X Máximo Promedio', linewidth=2)
    plt.xlabel('Generación')
    plt.ylabel('Posición X Máxima')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"graficas/metricas_adicionales_{config_name}.png", dpi=300, bbox_inches='tight')
    plt.show()

def graficar_comparacion_configuraciones():
    """Compara diferentes configuraciones en una misma gráfica"""
    resultados = cargar_datos()
    colores = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    
    for i, (config_name, datos) in enumerate(resultados.items()):
        color = colores[i % len(colores)]
        generaciones = range(len(datos['fitness_max']))

        #grafico para fitness maximo
        ax1.plot(generaciones, datos['fitness_max'], color=color,  label=config_name, linewidth=2)

        #grafico para fitness promedio
        ax2.plot(generaciones, datos['fitness_prom'], color=color,  label=config_name, linewidth=2)

    
    ax1.set_xlabel('Generación')
    ax1.set_ylabel('Fitness Máximo')
    ax1.set_title('Comparación del Fitness Máximo entre Configuraciones')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Generación')
    ax2.set_ylabel('Fitness Promedio')
    ax2.set_title('Comparación del Fitness Máximo entre Configuraciones')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout() #ajusta el espacioo de los subplots
    
    os.makedirs("graficas", exist_ok=True)
    plt.savefig("graficas/comparacion_configuraciones.png", dpi=300, bbox_inches='tight')
    plt.show()

def graficar_todas_metricas(config_name, datos):
    """Grafica todas las métricas en subplots"""
    generaciones = range(len(datos['fitness_max']))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Fitness
    ax1.plot(generaciones, datos['fitness_max'], 'b-', label='Máximo', linewidth=2)
    ax1.plot(generaciones, datos['fitness_prom'], 'r-', label='Promedio', linewidth=2)
    ax1.plot(generaciones, datos['fitness_min'], 'g-', label='Mínimo', alpha=0.7)
    ax1.set_title('Fitness por Generación')
    ax1.set_xlabel('Generación')
    ax1.set_ylabel('Fitness')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Recompensa promedio
    ax2.plot(generaciones, datos['recompensa_prom'], 'orange', linewidth=2)
    ax2.set_title('Recompensa Promedio')
    ax2.set_xlabel('Generación')
    ax2.set_ylabel('Recompensa')
    ax2.grid(True, alpha=0.3)
    
    # X máximo promedio
    ax3.plot(generaciones, datos['xmax_prom'], 'purple', linewidth=2)
    ax3.set_title('Posición X Máxima Promedio')
    ax3.set_xlabel('Generación')
    ax3.set_ylabel('Posición X')
    ax3.grid(True, alpha=0.3)
    
    # Fitness máximo con suavizado
    fitness_suavizado = gaussian_filter1d(datos['fitness_max'], sigma=1)
    ax4.plot(generaciones, datos['fitness_max'], 'b-', alpha=0.3, label='Original')
    ax4.plot(generaciones, fitness_suavizado, 'b-', linewidth=2, label='Suavizado')
    ax4.set_title('Fitness Máximo (Suavizado)')
    ax4.set_xlabel('Generación')
    ax4.set_ylabel('Fitness')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Análisis Completo - Configuración: {config_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"graficas/analisis_completo_{config_name}.png", dpi=300, bbox_inches='tight')
    plt.show()


def main():
    print("=== Visualizador de resultados ===")
    print("1. Graficar todas las configuraciones")
    print("2. Graficar configuración específica")
    print("3. Comparar configuraciones")
    print("4. Salir")
    
    while True:
        opcion = input("\nSelecciona una opción (1-4): ")
        
        if opcion == "1":
            resultados = cargar_datos()
            for config_name, datos in resultados.items():
                graficar_progreso(config_name, datos)
                graficar_todas_metricas(config_name, datos)
                
        elif opcion == "2":
            config_name = input("Ingresa el nombre de la configuración: ")
            try:
                datos = cargar_datos(config_name)
                graficar_progreso(config_name, datos)
                graficar_todas_metricas(config_name, datos)
            except FileNotFoundError:
                print(f"No se encontraron datos para la configuración: {config_name}")
                
        elif opcion == "3":
            graficar_comparacion_configuraciones()
            
        elif opcion == "4":
            break
            
        else:
            print("Opción no válida")

if __name__ == "__main__":
    main()