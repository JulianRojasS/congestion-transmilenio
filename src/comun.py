"""
Funciones compartidas del proyecto: carga y limpieza de los datos de
TransMilenio, construcción de variables y utilidades del árbol de decisión.

Fuente de datos: "Consolidado de Salidas Sistema Troncal por Franja Horaria",
TRANSMILENIO S.A., Datos Abiertos Bogotá (licencia CC BY 4.0).
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
RUTA_CRUDO = RAIZ / "data" / "raw" / "salidas_troncal_abril_2024.csv"
RUTA_PROCESADO = RAIZ / "data" / "processed" / "salidas_por_hora.csv"
RUTA_MODELO = RAIZ / "modelos" / "arbol_congestion.joblib"
RUTA_RESULTADOS = RAIZ / "resultados"
RUTA_FIGURAS = RUTA_RESULTADOS / "figuras"

# ---------------------------------------------------------------------------
# Parámetros del problema
# ---------------------------------------------------------------------------
ANIO, MES = 2024, 4                 # abril de 2024
DIAS_DEL_MES = 30                   # abril tiene 30 días (la columna DÍA 31 viene con "X")
HORA_INICIO, HORA_FIN = 5, 22       # horas de operación que se modelan (05:00 a 22:59)
ULTIMO_DIA_ENTRENAMIENTO = 23       # días 1-23 entrenan, días 24-30 (una semana) prueban
FESTIVOS = set()                    # en abril de 2024 no hubo festivos en Colombia

CLASES = ["bajo", "medio", "alto"]  # niveles de congestión
NOMBRES_DIA = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
TIPOS_DIA = ["hábil", "sábado", "domingo/festivo"]

# Filas del archivo que no son estaciones de pasajeros
PATRON_EXCLUIR = re.compile(r"corral|bicicletero", re.IGNORECASE)


# ---------------------------------------------------------------------------
# 1. Carga y limpieza
# ---------------------------------------------------------------------------
def columnas_dia() -> list[str]:
    return [f"DÍA {d:02d}" for d in range(1, DIAS_DEL_MES + 1)]


def a_entero(serie: pd.Series) -> pd.Series:
    """Convierte '1.026' -> 1026. El archivo usa punto como separador de miles."""
    limpio = serie.astype(str).str.strip().str.replace(".", "", regex=False)
    return pd.to_numeric(limpio, errors="raise").astype(int)


def separar_codigo(texto: str) -> tuple[str, str]:
    """'(09122) Calle 72' -> ('09122', 'Calle 72')."""
    m = re.match(r"^\((\w+)\)\s*(.+)$", str(texto).strip())
    if not m:
        return "", str(texto).strip()
    return m.group(1), m.group(2).strip()


def cargar_crudo(ruta: Path = RUTA_CRUDO) -> pd.DataFrame:
    """Lee el CSV original (separador ';', codificación Windows-1252)."""
    return pd.read_csv(ruta, sep=";", encoding="cp1252", dtype=str)


def limpiar(crudo: pd.DataFrame) -> pd.DataFrame:
    """
    - Quita las filas vacías del final del archivo.
    - Quita corrales y biciparqueaderos (no son estaciones de pasajeros).
    - Convierte las columnas de días a enteros (quitando el separador de miles).
    - Verifica que la suma de los días coincida con la columna 'Total general'.
    """
    df = crudo.dropna(subset=["Línea", "Estación"]).copy()
    df = df[~df["Estación"].str.contains(PATRON_EXCLUIR)]

    dias = columnas_dia()
    for c in dias:
        df[c] = a_entero(df[c])
    total = a_entero(df["Total general"])
    if not (df[dias].sum(axis=1) == total).all():
        raise ValueError("La suma de los días no coincide con 'Total general'")

    df["codigo_estacion"], df["estacion"] = zip(*df["Estación"].map(separar_codigo))
    df["codigo_zona"], df["zona"] = zip(*df["Línea"].map(separar_codigo))
    df["acceso"] = df["Acceso de Estación"].str.strip()
    df["hora"] = df["INTERVALO"].str[:2].astype(int)
    return df[["codigo_zona", "zona", "codigo_estacion", "estacion", "acceso",
               "INTERVALO", "hora"] + dias]


def tipo_dia(d: date, festivos: set[date] = FESTIVOS) -> int:
    """0 = hábil, 1 = sábado, 2 = domingo o festivo."""
    if d in festivos or d.weekday() == 6:
        return 2
    return 1 if d.weekday() == 5 else 0


def agregar_por_hora(limpio: pd.DataFrame) -> pd.DataFrame:
    """
    Pasa de formato ancho (una columna por día, una fila por acceso e intervalo
    de 15 min) a formato largo, y suma las salidas de todos los accesos de cada
    estación por fecha y hora. Resultado: una fila por estación, día y hora.
    """
    dias = columnas_dia()
    largo = limpio.melt(
        id_vars=["codigo_zona", "zona", "codigo_estacion", "estacion", "hora"],
        value_vars=dias, var_name="columna_dia", value_name="salidas")
    largo["dia"] = largo["columna_dia"].str[-2:].astype(int)

    por_hora = (largo.groupby(["codigo_zona", "zona", "codigo_estacion", "estacion",
                               "dia", "hora"], as_index=False)["salidas"].sum())

    # Rejilla completa: si una estación no tiene registros en una hora, son 0 salidas
    estaciones = por_hora[["codigo_zona", "zona", "codigo_estacion", "estacion"]].drop_duplicates()
    rejilla = estaciones.merge(
        pd.DataFrame({"dia": range(1, DIAS_DEL_MES + 1)}), how="cross").merge(
        pd.DataFrame({"hora": range(HORA_INICIO, HORA_FIN + 1)}), how="cross")
    por_hora = rejilla.merge(por_hora, how="left",
                             on=["codigo_zona", "zona", "codigo_estacion", "estacion", "dia", "hora"])
    por_hora["salidas"] = por_hora["salidas"].fillna(0).astype(int)

    # Variables de calendario
    fechas = por_hora["dia"].map(lambda d: date(ANIO, MES, d))
    por_hora["fecha"] = fechas.map(date.isoformat)
    por_hora["dia_semana"] = fechas.map(lambda f: f.weekday())
    por_hora["tipo_dia"] = fechas.map(tipo_dia)

    # Variables estructurales de la estación (no dependen de las salidas)
    accesos = limpio.groupby("codigo_estacion")["acceso"].nunique()
    por_hora["num_accesos"] = por_hora["codigo_estacion"].map(accesos).astype(int)
    por_hora["es_portal"] = por_hora["estacion"].str.contains(
        r"portal|cabecera", case=False).astype(int)

    columnas = ["fecha", "dia", "dia_semana", "tipo_dia", "hora", "codigo_zona", "zona",
                "codigo_estacion", "estacion", "num_accesos", "es_portal", "salidas"]
    return por_hora[columnas].sort_values(["codigo_estacion", "dia", "hora"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Variable objetivo y matriz de variables
# ---------------------------------------------------------------------------
def calcular_umbrales(salidas: pd.Series) -> tuple[float, float]:
    """Terciles de las salidas por hora (se calculan SOLO con entrenamiento)."""
    q1, q2 = salidas.quantile([1 / 3, 2 / 3])
    return float(q1), float(q2)


def etiquetar(salidas: pd.Series, umbrales: tuple[float, float]) -> np.ndarray:
    """bajo si <= q1, medio si <= q2, alto en otro caso."""
    q1, q2 = umbrales
    idx = np.select([salidas <= q1, salidas <= q2], [0, 1], 2)
    return np.array(CLASES, dtype=object)[idx]


def demanda_media_por_estacion(entrenamiento: pd.DataFrame) -> dict[str, float]:
    """Salidas promedio por hora de cada estación, calculadas con el periodo de entrenamiento."""
    return entrenamiento.groupby("codigo_estacion")["salidas"].mean().round(2).to_dict()


VARIABLES_NUMERICAS = ["hora", "dia_semana", "tipo_dia", "es_portal", "num_accesos", "demanda_media"]


def construir_X(df: pd.DataFrame, demanda_media: dict[str, float], zonas: list[str]) -> pd.DataFrame:
    """Matriz de variables: numéricas + zona troncal en one-hot."""
    X = df[["hora", "dia_semana", "tipo_dia", "es_portal", "num_accesos"]].copy()
    media_global = float(np.mean(list(demanda_media.values())))
    X["demanda_media"] = df["codigo_estacion"].map(demanda_media).fillna(media_global)
    for z in zonas:
        X[f"zona_{z}"] = (df["zona"] == z).astype(int)
    return X


# ---------------------------------------------------------------------------
# 3. Conceptos del capítulo 17: entropía y ganancia de información
# ---------------------------------------------------------------------------
def entropia(y) -> float:
    """H(S) = -sum p_i * log2(p_i)."""
    _, cuentas = np.unique(np.asarray(y), return_counts=True)
    p = cuentas / cuentas.sum()
    return float(-(p * np.log2(p)).sum())


def ganancia_informacion(y, mascara_izquierda) -> float:
    """G(S, A) = H(S) - sum |Sv|/|S| * H(Sv) para una partición binaria."""
    y = np.asarray(y)
    m = np.asarray(mascara_izquierda, dtype=bool)
    n = len(y)
    if m.all() or (~m).all():
        return 0.0
    return entropia(y) - (m.sum() / n) * entropia(y[m]) - ((~m).sum() / n) * entropia(y[~m])


# ---------------------------------------------------------------------------
# 4. Utilidades de texto para la búsqueda de estaciones
# ---------------------------------------------------------------------------
def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes ni signos: 'Héroes' -> 'heroes'."""
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


# Nombres populares que en el archivo aparecen con otro nombre
ALIAS_ESTACIONES = {
    "portal norte": "Cabecera Autopista Norte",
    "portal usme": "Cabecera Usme",
    "portal 80": "Cabecera Calle 80",
    "portal calle 80": "Cabecera Calle 80",
    "portal 20 de julio": "Portal 20 de Julio",
    "portal sur": "Portal Sur JFK Coop. Financiera",
    "heroes": "Heroes - Gel hada",
    "calle 100": "Calle 100 - Marketmedios",
    "jimenez": "Avenida Jimenez",
    "av jimenez": "Avenida Jimenez",
}
