"""Pruebas del modelo entrenado y del script de predicción."""
from datetime import date

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from comun import (CLASES, ULTIMO_DIA_ENTRENAMIENTO, construir_X, etiquetar)
from predecir import ErrorEntrada, buscar_estacion, explicar, main, predecir, validar_hora


@pytest.fixture(scope="module")
def prueba(procesado, paquete):
    df = procesado[procesado["dia"] > ULTIMO_DIA_ENTRENAMIENTO]
    X = construir_X(df, paquete["demanda_media"], paquete["zonas"])[paquete["columnas"]]
    y = etiquetar(df["salidas"], paquete["umbrales"])
    return df, X, y


# --- Calidad del modelo -------------------------------------------------------------
def test_desempeno_en_semana_de_prueba(prueba, paquete):
    _, X, y = prueba
    pred = paquete["modelo"].predict(X)
    assert f1_score(y, pred, average="macro") >= 0.85
    assert accuracy_score(y, pred) >= 0.85


def test_supera_ampliamente_a_la_linea_base(prueba, paquete):
    _, X, y = prueba
    base = max(np.mean(y == c) for c in CLASES)        # predecir siempre la clase mayoritaria
    assert accuracy_score(y, paquete["modelo"].predict(X)) > base + 0.4


def test_arbol_explicativo_es_pequeno_y_util(prueba, paquete):
    _, X, y = prueba
    arbol = paquete["explicativo"]
    assert arbol.get_depth() <= 4 and arbol.get_n_leaves() <= 16
    assert accuracy_score(y, arbol.predict(X)) >= 0.6


def test_sin_fuga_de_informacion(procesado, paquete):
    """La demanda media de cada estación se calculó solo con los días de entrenamiento."""
    ent = procesado[procesado["dia"] <= ULTIMO_DIA_ENTRENAMIENTO]
    esperado = ent.groupby("codigo_estacion")["salidas"].mean()
    for codigo in ["09122", "12003", "02000"]:
        assert paquete["demanda_media"][codigo] == pytest.approx(esperado[codigo], abs=0.01)
    q1, q2 = paquete["umbrales"]
    assert q1 == pytest.approx(ent["salidas"].quantile(1 / 3))
    assert q2 == pytest.approx(ent["salidas"].quantile(2 / 3))


# --- Predicción ----------------------------------------------------------------------
def test_prediccion_devuelve_clases_y_probabilidades_validas(paquete):
    res = predecir(paquete, "12003", date(2024, 4, 24), range(5, 23))
    assert len(res) == 18
    assert set(res["nivel"]) <= set(CLASES)
    assert np.allclose(res[["p_bajo", "p_medio", "p_alto"]].sum(axis=1), 1.0)


def test_casos_de_sentido_comun(paquete):
    # Estación grande en hora pico de un día hábil -> alto
    pico = predecir(paquete, buscar_estacion(paquete, "Calle 100"), date(2024, 4, 25), [8])
    assert pico["nivel"].iloc[0] == "alto"
    # Estación pequeña un domingo por la noche -> bajo
    noche = predecir(paquete, buscar_estacion(paquete, "Bicentenario"), date(2024, 4, 28), [21])
    assert noche["nivel"].iloc[0] == "bajo"


def test_festivo_se_trata_como_domingo(paquete):
    codigo = buscar_estacion(paquete, "Calle 72")
    festivo = predecir(paquete, codigo, date(2024, 4, 24), range(5, 23), festivo=True)
    domingo = predecir(paquete, codigo, date(2024, 4, 28), range(5, 23))
    habil = predecir(paquete, codigo, date(2024, 4, 24), range(5, 23))
    assert festivo["nivel"].tolist() == domingo["nivel"].tolist()
    assert festivo["nivel"].tolist() != habil["nivel"].tolist()


def test_explicacion_coincide_con_arbol_explicativo(paquete):
    codigo = "09122"
    res = predecir(paquete, codigo, date(2024, 4, 29), [7])
    regla = res["regla_explicativa"].iloc[0]
    filas = res[["hora"]].assign(dia_semana=0, tipo_dia=0, codigo_estacion=codigo,
                                 zona=paquete["estaciones"][codigo]["zona"],
                                 es_portal=0, num_accesos=paquete["estaciones"][codigo]["num_accesos"])
    X = construir_X(filas, paquete["demanda_media"], paquete["zonas"])[paquete["columnas"]]
    assert regla.startswith("SI ")
    assert regla.endswith(f"ENTONCES {paquete['explicativo'].predict(X)[0]}")
    assert explicar(paquete, X) == regla


# --- Búsqueda de estaciones y validaciones -----------------------------------------
@pytest.mark.parametrize("consulta, nombre", [
    ("09122", "Calle 72"),                         # por código
    ("calle 72", "Calle 72"),                      # sin mayúsculas
    ("HEROES", "Heroes - Gel hada"),               # alias / sin tilde
    ("portal norte", "Cabecera Autopista Norte"),  # nombre popular
    ("Ricuarte", "Ricaurte"),                      # error de digitación
    ("museo del oro", "Museo del Oro"),
])
def test_buscar_estacion(paquete, consulta, nombre):
    assert paquete["estaciones"][buscar_estacion(paquete, consulta)]["estacion"] == nombre


@pytest.mark.parametrize("consulta", ["calle", "xyz123", "   "])
def test_buscar_estacion_ambigua_o_inexistente(paquete, consulta):
    with pytest.raises(ErrorEntrada):
        buscar_estacion(paquete, consulta)


@pytest.mark.parametrize("hora", [0, 4, 23, 30])
def test_hora_fuera_de_operacion(hora):
    with pytest.raises(ErrorEntrada):
        validar_hora(hora)


def test_cli_codigos_de_salida(paquete, capsys):
    assert main(["--estacion", "Calle 72", "--fecha", "2024-04-29", "--hora", "7"]) == 0
    assert "congestión ALTO" in capsys.readouterr().out
    assert main(["--estacion", "Calle 72", "--fecha", "29/04/2024", "--hora", "7"]) == 1
    assert main(["--estacion", "Calle 72", "--fecha", "2024-04-29"]) == 1
