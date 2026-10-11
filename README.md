# Congestión TransMilenio – Aprendizaje supervisado con árboles de decisión

Modelo de aprendizaje supervisado que predice el **nivel de congestión (bajo, medio o alto)** de cada estación troncal de TransMilenio, por hora, a partir de **datos abiertos reales** de TRANSMILENIO S.A. (abril de 2024).

Este proyecto continúa el sistema experto de rutas de las actividades anteriores. Ese sistema encuentra la mejor ruta entre dos estaciones, y este modelo agrega la capacidad de anticipar qué tan congestionada estará una estación.

**Curso:** Inteligencia Artificial – Actividad 5: métodos supervisados
**Referencia teórica:** Palma Méndez, J. T. (2008). *Inteligencia artificial: métodos, técnicas y aplicaciones*, cap. 17 "Aprendizaje de árboles y reglas de decisión". McGraw-Hill.

## Resultados principales

| Modelo | Exactitud en la semana de prueba |
|---|---:|
| Línea base (clase más frecuente) | 33,7 % |
| Árbol explicativo (profundidad 4, 16 reglas) | 67,1 % |
| **Árbol de decisión ajustado (entropía)** | **91,1 %** |
| Random Forest (referencia) | 91,6 % |

El modelo se entrena con los días 1 a 23 de abril y se evalúa con la semana del 24 al 30, que nunca ve durante el entrenamiento.

![Matriz de confusión](resultados/figuras/matriz_confusion.png)

## Estructura del repositorio

```
congestion-transmilenio/
├── data/
│   ├── raw/salidas_troncal_abril_2024.csv     ← fuente original de TransMilenio (sin modificar)
│   └── processed/salidas_por_hora.csv         ← dataset limpio: estación × día × hora
├── src/
│   ├── comun.py             ← limpieza, variables, entropía y ganancia de información
│   ├── graficas.py          ← estilo de las gráficas
│   ├── preparar_datos.py    ← paso 1: limpieza y transformación
│   ├── entrenar_modelo.py   ← paso 2: entrenamiento, evaluación, reglas y gráficas
│   └── predecir.py          ← paso 3: predicción por consola
├── tests/                   ← 42 pruebas automáticas (pytest)
├── resultados/              ← métricas, reglas SI-ENTONCES, predicciones y figuras
├── modelos/                 ← modelo entrenado (se genera con el paso 2)
└── docs/
    ├── DESCRIPCION_DATOS.md ← entregable 3: descripción de los datos
    ├── PRUEBAS.md           ← entregable 4: pruebas realizadas
    └── GUIA_EQUIPO.md       ← reparto de commits y guion del video
```

## Instalación

Requiere Python 3.10 o superior.

```bash
git clone <URL-del-repositorio>
cd congestion-transmilenio
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
# 1. Preparar los datos (lee data/raw y genera data/processed)
python src/preparar_datos.py

# 2. Entrenar y evaluar los modelos (genera modelos/ y resultados/), unos 30 segundos
python src/entrenar_modelo.py

# 3. Predecir
python src/predecir.py --estacion "Calle 72" --fecha 2024-04-29 --hora 7
python src/predecir.py --estacion ricaurte --fecha 2024-04-24 --dia-completo
python src/predecir.py --estacion "portal norte" --fecha 2024-05-01 --hora 18 --festivo
python src/predecir.py --listar

# 4. Ejecutar las pruebas
python -m pytest -v
```

Ejemplo de salida:

```
Estación: Calle 72 (09122) – Zona A Caracas
Fecha:    2024-04-29 (lunes, hábil)
Niveles:  bajo <= 253 < medio <= 603 < alto  (salidas por hora)

Hora 07:00 -> congestión ALTO
Probabilidades: bajo 0% | medio 0% | alto 100%
Explicación (árbol de profundidad 4): SI 470.9 < demanda_media <= 1119.7 Y hora <= 19.5 Y tipo_dia <= 1.5 ENTONCES alto
Valor real: 3,604 salidas -> alto (acierto; día del periodo de prueba)
```

Para fechas de abril de 2024 el programa también muestra el valor real, lo que permite comprobar la predicción. La búsqueda de estaciones acepta nombres sin tildes, nombres populares ("portal norte") y errores de digitación ("Ricuarte").

## Cómo funciona

1. **Datos.** Se usa el *Consolidado de Salidas Sistema Troncal por Franja Horaria*: salidas por acceso de cada estación, cada 15 minutos. Se eliminan las filas vacías, se corrige el separador de miles, se pasa de formato ancho (una columna por día) a largo y se suman las salidas por estación y hora. Ver [docs/DESCRIPCION_DATOS.md](docs/DESCRIPCION_DATOS.md).
2. **Clase.** El nivel de congestión se define con los terciles de las salidas por hora del periodo de entrenamiento: bajo ≤ 253, medio ≤ 603 y alto > 603.
3. **Variables.** Hora, día de la semana, tipo de día, zona troncal, número de accesos, si es portal y la demanda media de la estación (calculada solo con datos de entrenamiento).
4. **Modelo.** `DecisionTreeClassifier` con `criterion="entropy"`, el mismo criterio de ganancia de información de ID3/C4.5 del capítulo 17. Los hiperparámetros se eligen con validación cruzada agrupada por día. El script recalcula a mano la entropía y la ganancia de información de la primera división, y el valor coincide con el de scikit-learn.
5. **Reglas.** Un árbol de profundidad 4 se convierte en 16 reglas SI-ENTONCES con su soporte y su confianza (`resultados/reglas_arbol_explicativo.txt`).
6. **Pruebas.** Evaluación en una semana no vista, análisis de errores, 12 casos funcionales y 42 pruebas automáticas. Ver [docs/PRUEBAS.md](docs/PRUEBAS.md).

## Fuente de los datos y licencia

Datos: **TRANSMILENIO S.A.**, *Consolidado de Salidas Sistema Troncal por Franja Horaria – abril de 2024*, publicado en [Datos Abiertos Bogotá](https://datosabiertos.bogota.gov.co/dataset/consolidado-de-salidas-sistema-troncal-por-franja-horaria) bajo licencia [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).

## Integrantes

| Integrante | Responsabilidad |
|---|---|
| Julian Rojas | Datos: fuente, limpieza y transformación |
| Julian Rojas | Modelo: entrenamiento, evaluación y reglas |
| Julian Rojas | Predictor por consola |
| Julian Rojas | Pruebas y documentación |
