"""
Paso 2: entrenamiento y evaluación del modelo supervisado.

Problema: clasificar el nivel de congestión (bajo / medio / alto) de una
estación troncal de TransMilenio en una hora dada, según la hora, el día,
la zona troncal y características de la estación.

Modelos:
  - Línea base: predecir siempre la clase más frecuente.
  - Árbol de decisión (criterio de entropía, cap. 17 de Palma) con búsqueda
    de hiperparámetros por validación cruzada agrupada por día.
  - Árbol explicativo de profundidad 4, para leer sus reglas SI-ENTONCES.
  - Random Forest, como punto de comparación.

Validación: los días 1-23 de abril entrenan; los días 24-30 (una semana
completa, que nunca ve el modelo) se usan como prueba.

Uso:
    python src/entrenar_modelo.py
"""
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from comun import (CLASES, RUTA_FIGURAS, RUTA_MODELO, RUTA_PROCESADO, RUTA_RESULTADOS,
                   ULTIMO_DIA_ENTRENAMIENTO, calcular_umbrales, construir_X,
                   demanda_media_por_estacion, entropia, etiquetar, ganancia_informacion)
from graficas import (AZUL_SECUENCIAL, COLORES_NIVELES, COLORES_SERIES, TEXTO_PRIMARIO,
                      TEXTO_SECUNDARIO, estilo_base)

SEMILLA = 42
PROFUNDIDAD_EXPLICATIVA = 4
REJILLA_HIPERPARAMETROS = {
    "max_depth": [4, 6, 8, 10, 12, 15, 20, None],
    "min_samples_leaf": [1, 5, 10, 20],
}


# ---------------------------------------------------------------------------
def cargar_y_dividir():
    df = pd.read_csv(RUTA_PROCESADO, dtype={"codigo_estacion": str, "codigo_zona": str})
    entrenamiento = df[df["dia"] <= ULTIMO_DIA_ENTRENAMIENTO].reset_index(drop=True)
    prueba = df[df["dia"] > ULTIMO_DIA_ENTRENAMIENTO].reset_index(drop=True)
    return df, entrenamiento, prueba


def metricas(y_real, y_pred) -> dict:
    return {
        "exactitud": round(float(accuracy_score(y_real, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_real, y_pred, average="macro")), 4),
    }


def explicar_raiz(arbol, X, y) -> dict:
    """Recalcula a mano la entropía y la ganancia de información de la primera división."""
    t = arbol.tree_
    variable = X.columns[t.feature[0]]
    umbral = float(t.threshold[0])
    izquierda = X[variable].to_numpy() <= umbral
    return {
        "variable": variable,
        "umbral": round(umbral, 2),
        "entropia_conjunto": round(entropia(y), 4),
        "entropia_maxima_3_clases": round(float(np.log2(3)), 4),
        "entropia_izquierda": round(entropia(y[izquierda]), 4),
        "entropia_derecha": round(entropia(y[~izquierda]), 4),
        "ejemplos_izquierda": int(izquierda.sum()),
        "ejemplos_derecha": int((~izquierda).sum()),
        "ganancia_informacion_manual": round(ganancia_informacion(y, izquierda), 4),
        "ganancia_informacion_sklearn": round(
            float(t.impurity[0]
                  - t.weighted_n_node_samples[t.children_left[0]] / t.weighted_n_node_samples[0]
                  * t.impurity[t.children_left[0]]
                  - t.weighted_n_node_samples[t.children_right[0]] / t.weighted_n_node_samples[0]
                  * t.impurity[t.children_right[0]]), 4),
    }


def reglas_si_entonces(arbol, columnas) -> list[dict]:
    """
    Convierte cada camino raíz-hoja en una regla SI-ENTONCES (cap. 17: paso de
    árboles a reglas). Une las condiciones repetidas sobre la misma variable
    en un intervalo y calcula soporte y confianza de cada regla.
    """
    t = arbol.tree_
    total = t.weighted_n_node_samples[0]
    reglas = []

    def recorrer(nodo, limites):
        if t.children_left[nodo] == -1:  # hoja
            v = t.value[nodo][0]
            clase = arbol.classes_[int(np.argmax(v))]
            condiciones = []
            for var, (bajo, alto) in limites.items():
                if bajo is not None and alto is not None:
                    condiciones.append(f"{bajo:.1f} < {var} <= {alto:.1f}")
                elif alto is not None:
                    condiciones.append(f"{var} <= {alto:.1f}")
                else:
                    condiciones.append(f"{var} > {bajo:.1f}")
            reglas.append({"condiciones": condiciones, "clase": clase,
                           "soporte": float(t.weighted_n_node_samples[nodo] / total),
                           "confianza": float(v.max() / v.sum())})
            return
        var = columnas[t.feature[nodo]]
        umbral = float(t.threshold[nodo])
        bajo, alto = limites.get(var, (None, None))
        izq = dict(limites); izq[var] = (bajo, umbral if alto is None else min(alto, umbral))
        der = dict(limites); der[var] = (umbral if bajo is None else max(bajo, umbral), alto)
        recorrer(t.children_left[nodo], izq)
        recorrer(t.children_right[nodo], der)

    recorrer(0, {})
    return reglas


def texto_reglas(reglas) -> str:
    lineas = []
    for i, r in enumerate(reglas, 1):
        lineas.append(f"R{i:02d}: SI " + " Y ".join(r["condiciones"]) +
                      f" ENTONCES congestión = {r['clase'].upper()}"
                      f"   (soporte {r['soporte']:.1%}, confianza {r['confianza']:.1%})")
    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------
def grafica_matriz_confusion(cm, ruta):
    estilo_base()
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    porcentaje = cm / cm.sum(axis=1, keepdims=True)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("azul", AZUL_SECUENCIAL)
    ax.imshow(porcentaje, cmap=cmap, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(len(CLASES)):
        for j in range(len(CLASES)):
            claro = porcentaje[i, j] > 0.5
            ax.text(j, i, f"{cm[i, j]:,}\n{porcentaje[i, j]:.0%}", ha="center", va="center",
                    color="white" if claro else TEXTO_PRIMARIO, fontsize=10)
    ax.set_xticks(range(3), CLASES)
    ax.set_yticks(range(3), CLASES)
    ax.set_xlabel("Nivel predicho")
    ax.set_ylabel("Nivel real")
    ax.set_title("Matriz de confusión – semana de prueba (24-30 abr)", loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def grafica_comparacion(resultados, ruta):
    estilo_base()
    nombres = list(resultados)
    valores = [resultados[n]["prueba"]["f1_macro"] for n in nombres]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(nombres))[::-1]
    ax.barh(y, valores, height=0.55, color=COLORES_SERIES[0])
    for yi, v in zip(y, valores):
        ax.text(v + 0.01, yi, f"{v:.3f}", va="center", color=TEXTO_PRIMARIO, fontsize=10)
    ax.set_yticks(y, nombres)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 macro en la semana de prueba")
    ax.set_title("Comparación de modelos", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def grafica_importancia(modelo, columnas, ruta, top=10):
    estilo_base()
    imp = pd.Series(modelo.feature_importances_, index=columnas).sort_values(ascending=False)[:top]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(imp))[::-1]
    ax.barh(y, imp.values, height=0.6, color=COLORES_SERIES[0])
    for yi, v in zip(y, imp.values):
        ax.text(v + 0.005, yi, f"{v:.1%}", va="center", color=TEXTO_PRIMARIO, fontsize=9)
    ax.set_yticks(y, [c.replace("zona_", "zona: ") for c in imp.index])
    ax.set_xlabel("Importancia (reducción total de entropía)")
    ax.set_title("Variables más importantes del árbol de decisión", loc="left")
    ax.set_xlim(0, imp.max() * 1.15)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return {k: round(float(v), 4) for k, v in imp.items()}


def grafica_arbol(arbol, columnas, ruta):
    estilo_base()
    fig, ax = plt.subplots(figsize=(22, 10))
    nodos = plot_tree(arbol, feature_names=columnas, class_names=list(arbol.classes_),
                      filled=False, impurity=True, proportion=True, rounded=True,
                      precision=2, fontsize=8, ax=ax)
    # Colorea cada caja según la clase mayoritaria escrita en ella
    # (plot_tree también devuelve las etiquetas True/False, que no tienen clase)
    for caja in nodos:
        texto = caja.get_text()
        if "class = " not in texto:
            continue
        clase = texto.split("class = ")[-1].strip()
        borde = caja.get_bbox_patch()
        if borde is not None:
            borde.set_facecolor(COLORES_NIVELES[clase])
            borde.set_alpha(0.35)
            borde.set_edgecolor(TEXTO_SECUNDARIO)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=COLORES_NIVELES[c], alpha=0.35, edgecolor=TEXTO_SECUNDARIO,
                             label=f"nivel {c}") for c in CLASES],
              loc="upper right", frameon=False, fontsize=11)
    ax.set_title(f"Árbol explicativo (profundidad {arbol.get_depth()}) – color = nivel mayoritario"
                 "  ·  value = proporción [alto, bajo, medio]", loc="left", fontsize=14)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)

    df, entrenamiento, prueba = cargar_y_dividir()
    umbrales = calcular_umbrales(entrenamiento["salidas"])
    y_ent = etiquetar(entrenamiento["salidas"], umbrales)
    y_pru = etiquetar(prueba["salidas"], umbrales)

    demanda = demanda_media_por_estacion(entrenamiento)
    zonas = sorted(df["zona"].unique())
    X_ent = construir_X(entrenamiento, demanda, zonas)
    X_pru = construir_X(prueba, demanda, zonas)
    columnas = list(X_ent.columns)

    print(f"Entrenamiento: {len(X_ent):,} filas (días 1-{ULTIMO_DIA_ENTRENAMIENTO})")
    print(f"Prueba:        {len(X_pru):,} filas (días {ULTIMO_DIA_ENTRENAMIENTO + 1}-30)")
    print(f"Umbrales: bajo <= {umbrales[0]:.0f} < medio <= {umbrales[1]:.0f} < alto (salidas/hora)")
    print(f"Variables: {len(columnas)}")

    resultados = {}

    # 1. Línea base
    base = DummyClassifier(strategy="most_frequent").fit(X_ent, y_ent)
    resultados["Línea base (clase más frecuente)"] = {
        "prueba": metricas(y_pru, base.predict(X_pru))}

    # 2. Árbol explicativo de profundidad fija
    explicativo = DecisionTreeClassifier(criterion="entropy", max_depth=PROFUNDIDAD_EXPLICATIVA,
                                         min_samples_leaf=20, random_state=SEMILLA)
    explicativo.fit(X_ent, y_ent)
    resultados[f"Árbol explicativo (prof. {PROFUNDIDAD_EXPLICATIVA})"] = {
        "prueba": metricas(y_pru, explicativo.predict(X_pru)),
        "hojas": int(explicativo.get_n_leaves())}

    # 3. Árbol de decisión con búsqueda de hiperparámetros
    print("\nBuscando hiperparámetros (validación cruzada de 5 particiones agrupadas por día)...")
    t0 = time.time()
    busqueda = GridSearchCV(
        DecisionTreeClassifier(criterion="entropy", random_state=SEMILLA),
        REJILLA_HIPERPARAMETROS, scoring="f1_macro", cv=GroupKFold(n_splits=5), n_jobs=-1)
    busqueda.fit(X_ent, y_ent, groups=entrenamiento["dia"])
    arbol = busqueda.best_estimator_
    pred_arbol = arbol.predict(X_pru)
    resultados["Árbol de decisión (ajustado)"] = {
        "prueba": metricas(y_pru, pred_arbol),
        "entrenamiento": metricas(y_ent, arbol.predict(X_ent)),
        "f1_validacion_cruzada": round(float(busqueda.best_score_), 4),
        "mejores_hiperparametros": {k: v for k, v in busqueda.best_params_.items()},
        "profundidad": int(arbol.get_depth()),
        "hojas": int(arbol.get_n_leaves()),
    }
    print(f"  Mejor: {busqueda.best_params_}  F1 CV = {busqueda.best_score_:.4f}  "
          f"({time.time() - t0:.0f} s)")

    # 4. Random Forest
    bosque = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1,
                                    random_state=SEMILLA).fit(X_ent, y_ent)
    resultados["Random Forest (200 árboles)"] = {"prueba": metricas(y_pru, bosque.predict(X_pru))}

    # Resultados detallados del modelo principal
    cm = confusion_matrix(y_pru, pred_arbol, labels=CLASES)
    reporte = classification_report(y_pru, pred_arbol, labels=CLASES, digits=3)
    raiz = explicar_raiz(arbol, X_ent, y_ent)
    reglas = ("ÁRBOL EXPLICATIVO (profundidad 4) - estructura\n\n"
              + export_text(explicativo, feature_names=columnas, decimals=1, show_weights=False)
              + "\n\nREGLAS SI-ENTONCES EXTRAÍDAS DEL ÁRBOL\n"
              "(soporte = % de ejemplos de entrenamiento que cumplen la regla;"
              " confianza = % de esos ejemplos que tienen la clase indicada)\n\n"
              + texto_reglas(reglas_si_entonces(explicativo, columnas)) + "\n")

    grafica_matriz_confusion(cm, RUTA_FIGURAS / "matriz_confusion.png")
    grafica_comparacion(resultados, RUTA_FIGURAS / "comparacion_modelos.png")
    importancia = grafica_importancia(arbol, columnas, RUTA_FIGURAS / "importancia_variables.png")
    grafica_arbol(explicativo, columnas, RUTA_FIGURAS / "arbol_explicativo.png")

    # Predicciones de la semana de prueba (para el documento de pruebas)
    salida_pred = prueba[["fecha", "hora", "codigo_estacion", "estacion", "zona", "salidas"]].copy()
    salida_pred["nivel_real"] = y_pru
    salida_pred["nivel_predicho"] = pred_arbol
    salida_pred["acierto"] = salida_pred["nivel_real"] == salida_pred["nivel_predicho"]
    salida_pred.to_csv(RUTA_RESULTADOS / "predicciones_prueba.csv", index=False, encoding="utf-8")

    # Guardar todo
    reporte_json = {
        "division": {"entrenamiento_dias": f"1-{ULTIMO_DIA_ENTRENAMIENTO}",
                     "prueba_dias": f"{ULTIMO_DIA_ENTRENAMIENTO + 1}-30",
                     "filas_entrenamiento": len(X_ent), "filas_prueba": len(X_pru)},
        "umbrales_salidas_por_hora": {"bajo_hasta": umbrales[0], "medio_hasta": umbrales[1]},
        "distribucion_clases": {
            "entrenamiento": pd.Series(y_ent).value_counts().reindex(CLASES).astype(int).to_dict(),
            "prueba": pd.Series(y_pru).value_counts().reindex(CLASES).astype(int).to_dict()},
        "modelos": resultados,
        "matriz_confusion": {"etiquetas": CLASES, "valores": cm.tolist()},
        "primera_division": raiz,
        "importancia_variables": importancia,
    }
    with open(RUTA_RESULTADOS / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(reporte_json, f, ensure_ascii=False, indent=2)
    (RUTA_RESULTADOS / "reporte_clasificacion.txt").write_text(reporte, encoding="utf-8")
    (RUTA_RESULTADOS / "reglas_arbol_explicativo.txt").write_text(reglas, encoding="utf-8")

    estaciones = (df[["codigo_estacion", "estacion", "zona", "num_accesos", "es_portal"]]
                  .drop_duplicates("codigo_estacion").set_index("codigo_estacion")
                  .to_dict(orient="index"))
    joblib.dump({"modelo": arbol, "explicativo": explicativo, "columnas": columnas,
                 "zonas": zonas, "umbrales": umbrales, "demanda_media": demanda,
                 "estaciones": estaciones}, RUTA_MODELO, compress=3)

    # Resumen en consola
    print("\nResultados en la semana de prueba:")
    for nombre, r in resultados.items():
        print(f"  {nombre:<36} exactitud {r['prueba']['exactitud']:.3f}   "
              f"F1 macro {r['prueba']['f1_macro']:.3f}")
    print("\nReporte del árbol de decisión ajustado:\n" + reporte)
    print("Primera división del árbol (cálculo manual vs. sklearn):")
    print(f"  {raiz['variable']} <= {raiz['umbral']}   H(S) = {raiz['entropia_conjunto']}   "
          f"ganancia manual = {raiz['ganancia_informacion_manual']}   "
          f"sklearn = {raiz['ganancia_informacion_sklearn']}")
    print(f"\nModelo guardado en {RUTA_MODELO}")


if __name__ == "__main__":
    main()
