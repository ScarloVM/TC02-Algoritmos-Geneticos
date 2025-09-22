CONFIGURACIONES = {
    'config1': {
        'POBLACION': 30,
        'GENERACIONES': 200,
        'ELITE': 2,
        'TASA_MUTACION': 0.10,
        'DESV_MUTACION': 0.15,
        'EPISODIOS_EVAL': 8,
        'PASOS_MAX': 200
    },
    'config2': {
        'POBLACION': 100,
        'GENERACIONES': 150,
        'ELITE': 8,
        'TASA_MUTACION': 0.35,
        'DESV_MUTACION': 0.30,
        'EPISODIOS_EVAL': 3,
        'PASOS_MAX': 200
    },
    'config3': {
        'POBLACION': 60, # tamaño de la población
        'GENERACIONES': 100, # número de generaciones
        'ELITE': 4, # cuántos mejores se copian directo
        'TASA_MUTACION': 0.20, # prob. de mutar cada gen
        'DESV_MUTACION': 0.20, # sigma del ruido gaussiano de mutación
        'EPISODIOS_EVAL': 5, # episodios por individuo (promedia)
        'PASOS_MAX': 200 # pasos máx. por episodio (bucle interno)
    }
}