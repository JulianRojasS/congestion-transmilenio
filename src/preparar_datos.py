"""
Paso 1: preparación de los datos.

Lee el CSV original de TransMilenio (salidas por acceso cada 15 minutos),
lo limpia, lo transforma a una fila por estación-día-hora y guarda
data/processed/salidas_por_hora.csv. También genera un resumen y una
gráfica exploratoria del perfil horario.

Uso:
    python src/preparar_datos.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from comun import (RUTA_CRUDO, RUTA_FIGURAS, RUTA_PROCESADO, RUTA_RESULTADOS, TIPOS_DIA,
                   agregar_por_hora, cargar_crudo, columnas_dia, limpiar)
from graficas import estilo_base, COLORES_SERIES, TEXTO_SECUNDARIO


def resumen(crudo, limpio, por_hora) -> dict:
    return {
        "archivo_original": RUTA_CRUDO.name,
        "filas_originales": int(len(crudo)),
        "filas_vacias_eliminadas": int(crudo["Línea"].isna().sum()),
        "filas_limpias": int(len(limpio)),
        "zonas_troncales": int(limpio["codigo_zona"].nunique()),
        "estaciones": int(limpio["codigo_estacion"].nunique()),
        "accesos": int(limpio["acceso"].nunique()),
        "intervalos_15min": int(limpio["INTERVALO"].nunique()),
        "salidas_totales_mes": int(limpio[columnas_dia()].to_numpy().sum()),
        "filas_dataset_por_hora": int(len(por_hora)),
        "salidas_en_horas_modeladas": int(por_hora["salidas"].sum()),
        "salidas_por_hora": {k: round(float(v), 1) for k, v in
                             por_hora["salidas"].describe().items()},
    }


def grafica_perfil_horario(por_hora):
    """Promedio de salidas por estación y hora, para cada tipo de día."""
    estilo_base()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    perfil = por_hora.groupby(["tipo_dia", "hora"])["salidas"].mean().unstack(0)
    for i, t in enumerate(perfil.columns):
        ax.plot(perfil.index, perfil[t], color=COLORES_SERIES[i], lw=2,
                marker="o", ms=4, label=TIPOS_DIA[t])
        ax.annotate(TIPOS_DIA[t], (perfil.index[-1], perfil[t].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    color=TEXTO_SECUNDARIO, fontsize=9)
    ax.set_xticks(range(5, 23))
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Salidas promedio por estación")
    ax.set_title("Perfil horario de salidas – TransMilenio troncal, abril 2024", loc="left")
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlim(4.5, 23.8)
    fig.tight_layout()
    ruta = RUTA_FIGURAS / "perfil_horario.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def main():
    print("Leyendo", RUTA_CRUDO)
    crudo = cargar_crudo()
    limpio = limpiar(crudo)
    por_hora = agregar_por_hora(limpio)

    RUTA_PROCESADO.parent.mkdir(parents=True, exist_ok=True)
    RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    por_hora.to_csv(RUTA_PROCESADO, index=False, encoding="utf-8")

    info = resumen(crudo, limpio, por_hora)
    with open(RUTA_RESULTADOS / "resumen_datos.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    ruta_fig = grafica_perfil_horario(por_hora)

    print(f"Filas originales:          {info['filas_originales']:,}")
    print(f"Filas vacías eliminadas:   {info['filas_vacias_eliminadas']:,}")
    print(f"Estaciones:                {info['estaciones']}  |  Zonas: {info['zonas_troncales']}")
    print(f"Salidas totales del mes:   {info['salidas_totales_mes']:,}")
    print(f"Dataset por hora:          {info['filas_dataset_por_hora']:,} filas -> {RUTA_PROCESADO}")
    print(f"Gráfica:                   {ruta_fig}")


if __name__ == "__main__":
    main()
