"""Pruebas de la preparación de datos y de las funciones del capítulo 17."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from comun import (CLASES, DIAS_DEL_MES, HORA_FIN, HORA_INICIO, a_entero, calcular_umbrales,
                   columnas_dia, entropia, etiquetar, ganancia_informacion, limpiar, normalizar,
                   separar_codigo, tipo_dia)


# --- Funciones unitarias --------------------------------------------------------
def test_separador_de_miles():
    assert a_entero(pd.Series(["0", "45", "1.026", "12.345"])).tolist() == [0, 45, 1026, 12345]


def test_separar_codigo_y_nombre():
    assert separar_codigo("(09122) Calle 72") == ("09122", "Calle 72")
    assert separar_codigo("(34) Zona H Caracas Sur") == ("34", "Zona H Caracas Sur")


@pytest.mark.parametrize("dia, esperado", [(1, 0), (5, 0), (6, 1), (7, 2), (8, 0)])
def test_tipo_de_dia_abril_2024(dia, esperado):
    # El 1 de abril de 2024 fue lunes
    assert tipo_dia(date(2024, 4, dia)) == esperado


def test_festivo_cuenta_como_domingo():
    assert tipo_dia(date(2024, 5, 1), festivos={date(2024, 5, 1)}) == 2


def test_normalizar_quita_tildes_y_signos():
    assert normalizar("Héroes - Gel hada") == "heroes gel hada"


def test_limpiar_elimina_vacias_corrales_y_valida_total():
    dias = columnas_dia()
    base = {"Línea": "(11) Zona K Calle 26", "Acceso de Estación": "(01) Acceso", "MES": "ABRIL",
            "INTERVALO": "07:15", **{d: "1" for d in dias}, "Total general": str(DIAS_DEL_MES)}
    filas = [dict(base, **{"Estación": "(06001) Modelia"}),
             dict(base, **{"Estación": "(50008) Corral Portal Dorado"}),
             {k: np.nan for k in base} | {"Estación": np.nan}]
    df = limpiar(pd.DataFrame(filas))
    assert df["estacion"].tolist() == ["Modelia"]
    assert df["hora"].tolist() == [7]

    malo = pd.DataFrame([dict(base, **{"Estación": "(06001) Modelia", "Total general": "999"})])
    with pytest.raises(ValueError):
        limpiar(malo)


def test_etiquetar_por_umbrales():
    y = etiquetar(pd.Series([0, 100, 101, 200, 201]), (100, 200))
    assert y.tolist() == ["bajo", "bajo", "medio", "medio", "alto"]


def test_umbrales_dan_clases_balanceadas():
    s = pd.Series(np.arange(3000))
    y = etiquetar(s, calcular_umbrales(s))
    conteo = pd.Series(y).value_counts()
    assert set(conteo.index) == set(CLASES)
    assert conteo.max() - conteo.min() <= 2


# --- Entropía y ganancia de información (cap. 17) -------------------------------
def test_entropia_casos_conocidos():
    assert entropia(["a", "a", "a"]) == 0
    assert entropia(["a", "b"]) == pytest.approx(1.0)
    assert entropia(["a", "b", "c"]) == pytest.approx(np.log2(3))


def test_ganancia_division_perfecta_y_nula():
    y = np.array(["a", "a", "b", "b"])
    assert ganancia_informacion(y, [True, True, False, False]) == pytest.approx(1.0)
    assert ganancia_informacion(y, [True, False, True, False]) == pytest.approx(0.0)
    assert ganancia_informacion(y, [True, True, True, True]) == 0.0


# --- Datos reales -----------------------------------------------------------------
def test_archivo_original_tiene_estructura_esperada(crudo):
    for c in ["Línea", "Estación", "Acceso de Estación", "INTERVALO", "DÍA 01", "DÍA 30",
              "Total general"]:
        assert c in crudo.columns
    assert (crudo["DÍA 31"].dropna() == "X").all()   # abril no tiene día 31


def test_limpieza_conserva_las_salidas(limpio, crudo):
    validas = crudo.dropna(subset=["Línea"])
    esperado = a_entero(validas["Total general"]).sum()
    excluidas = validas["Estación"].str.contains("Corral|Bicicletero", case=False)
    assert limpio[columnas_dia()].to_numpy().sum() == \
        esperado - a_entero(validas.loc[excluidas, "Total general"]).sum()


def test_dataset_por_hora_completo_y_sin_duplicados(por_hora):
    estaciones = por_hora["codigo_estacion"].nunique()
    horas = HORA_FIN - HORA_INICIO + 1
    assert len(por_hora) == estaciones * DIAS_DEL_MES * horas
    assert not por_hora.duplicated(["codigo_estacion", "dia", "hora"]).any()
    assert (por_hora["salidas"] >= 0).all()
    assert por_hora["hora"].between(HORA_INICIO, HORA_FIN).all()


def test_agregacion_por_hora_coincide_con_original(limpio, por_hora):
    """La suma por estación y día debe coincidir con los registros de 15 minutos."""
    dentro = limpio[limpio["hora"].between(HORA_INICIO, HORA_FIN)]
    esperado = dentro.groupby("codigo_estacion")[columnas_dia()].sum().sum(axis=1)
    obtenido = por_hora.groupby("codigo_estacion")["salidas"].sum()
    pd.testing.assert_series_equal(obtenido.sort_index(), esperado.sort_index(),
                                   check_names=False)


def test_calendario_correcto(por_hora):
    lunes = por_hora[por_hora["fecha"] == "2024-04-01"]
    assert (lunes["dia_semana"] == 0).all() and (lunes["tipo_dia"] == 0).all()
    domingo = por_hora[por_hora["fecha"] == "2024-04-07"]
    assert (domingo["tipo_dia"] == 2).all()


def test_archivo_procesado_coincide_con_regenerado(procesado, por_hora):
    assert len(procesado) == len(por_hora)
    assert procesado["salidas"].sum() == por_hora["salidas"].sum()
